import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, diagnose, type EvalResult, type RunEvent, type RunSummary } from "./api";
import { renderMarkdown } from "./markdown";
import ChatPanel from "./ChatPanel";
import "./styles.css";

type PlanStep = {
  id?: string;
  kind?: string;
  tool?: string;
  args?: Record<string, unknown>;
  description?: string;
};

type Live = {
  sessionId: string;
  runId: string | null;
  plan: PlanStep[];
  completed: Set<string>;
  replaced: Record<string, PlanStep[]>;
  events: RunEvent[];
  report: string;
  status: "idle" | "running" | "done" | "error";
  error: string;
  backend: "local" | "cloud" | null;
};

const emptyLive = (sessionId: string): Live => ({
  sessionId,
  runId: null,
  plan: [],
  completed: new Set(),
  replaced: {},
  events: [],
  report: "",
  status: "idle",
  error: "",
  backend: null,
});

function newSessionId() {
  return `web-${Date.now().toString(36)}`;
}

function fmtTime(value: string | null | undefined) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleTimeString("zh-CN", { hour12: false });
}

function fmtDuration(ms: number | null | undefined) {
  if (ms == null) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

function shortId(id: string) {
  return id.length > 10 ? `${id.slice(0, 6)}…${id.slice(-3)}` : id;
}

export default function App() {
  const [sessionId, setSessionId] = useState(newSessionId);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [evals, setEvals] = useState<EvalResult[]>([]);
  const [live, setLive] = useState<Live>(() => emptyLive(sessionId));
  const [selected, setSelected] = useState<string | null>(null);
  const [selectedEvents, setSelectedEvents] = useState<RunEvent[]>([]);
  const [notice, setNotice] = useState("");
  const [mode, setMode] = useState<"diagnose" | "chat">("diagnose");
  const abortRef = useRef<AbortController | null>(null);
  const latestRunId = useRef<string | null>(null);
  const streamEnd = useRef<HTMLDivElement | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [runData, evalData] = await Promise.all([
        api.listRuns(),
        api.listEvalResults(),
      ]);
      setRuns(runData.runs);
      setEvals(evalData.results);
      setNotice("");
    } catch (error) {
      setNotice(
        `无法加载运行记录：${(error as Error).message}。确认后端已启动（默认 http://127.0.0.1:9900）。`,
      );
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (live.status !== "running") return;
    streamEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [live.events.length, live.status]);

  const start = async () => {
    if (live.status === "running") return;
    const id = sessionId || newSessionId();
    setSessionId(id);
    setSelected(null);
    setSelectedEvents([]);
    setLive({ ...emptyLive(id), status: "running" });

    const controller = new AbortController();
    abortRef.current = controller;

    await diagnose(
      id,
      {
        onEvent: (event) => {
          if (event.run_id) latestRunId.current = event.run_id;
          setLive((prev) => applyEvent(prev, event));
        },
        onDone: () => {
          setLive((prev) => ({ ...prev, status: "done" }));
          void backfill(id);
        },
        onError: (message) => {
          setLive((prev) => ({ ...prev, status: "error", error: message }));
          void refresh();
        },
      },
      controller.signal,
    );
  };

  /** SSE 只带过程事件；完成后用 trace 补齐 token、耗时和节点明细。 */
  const backfill = async (sessionIdToMatch: string) => {
    await refresh();
    try {
      const { runs: latest } = await api.listRuns();
      const match =
        latest.find((run) => run.session_id === sessionIdToMatch) ?? latest[0];
      if (!match) return;
      const data = await api.getRun(match.run_id);
      setLive((prev) => ({
        ...replay(prev.sessionId, data.events),
        status: "done",
      }));
    } catch {
      // trace 还没落盘时保持 SSE 结果
    }
  };

  const openRun = async (runId: string) => {
    setSelected(runId);
    try {
      const data = await api.getRun(runId);
      setSelectedEvents(data.events);
    } catch (error) {
      setNotice(`读取 run 失败：${(error as Error).message}`);
    }
  };

  const current = useMemo(() => {
    if (!selected) return live;
    return replay(live.sessionId, selectedEvents);
  }, [selected, selectedEvents, live]);

  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand">
          <h1>On-Call Agent</h1>
          <p>plan · execute · verify · replan</p>
        </div>

        <div className="rail-actions">
          <button className="btn" onClick={() => void start()} disabled={live.status === "running"}>
            {live.status === "running" ? "诊断中…" : "开始诊断"}
          </button>
          <button
            className="btn ghost"
            onClick={() => {
              const id = newSessionId();
              setSessionId(id);
              setLive(emptyLive(id));
              setSelected(null);
              setSelectedEvents([]);
            }}
          >
            新会话
          </button>
        </div>

        <div className="session-id">session {sessionId}</div>

        <div className="runs">
          <InspectorHeading text={`最近运行 · ${runs.length}`} />
          {runs.length === 0 && (
            <p className="empty">
              还没有运行记录。点“开始诊断”跑一次，这里会出现每次运行的耗时、步数和 token。
            </p>
          )}
          {runs.map((run) => (
            <button
              key={run.run_id}
              className={`run-item ${run.status === "error" ? "error" : ""}`}
              aria-current={selected === run.run_id}
              onClick={() => void openRun(run.run_id)}
            >
              <span className="run-time">{fmtTime(run.started_at)}</span>
              <span className="run-time">{fmtDuration(run.duration_ms)}</span>
              <span className="run-meta">
                {run.steps} 步 · {run.backend === "local" ? "本地" : "云端"} ·{" "}
                {run.severity || "—"} · {run.category || "—"} ·{" "}
                {run.input_tokens + run.output_tokens} tok
              </span>
            </button>
          ))}
        </div>
      </aside>

      <main className="stage">
        <header className="stage-head">
          <div>
            <h2>
              {mode === "chat"
                ? "知识库问答"
                : selected
                  ? "回放运行"
                  : "现场诊断"}
            </h2>
            <span className="sub">
              {mode === "chat"
                ? "RAG + MCP tools"
                : selected
                  ? `run ${shortId(selected)}`
                  : `session ${shortId(live.sessionId)}`}
            </span>
          </div>
          <div className="mode-switch" role="group" aria-label="切换视图">
            <button
              aria-pressed={mode === "diagnose"}
              onClick={() => setMode("diagnose")}
            >
              诊断
            </button>
            <button aria-pressed={mode === "chat"} onClick={() => setMode("chat")}>
              问答
            </button>
          </div>
          <span className="sub">
            {current.status === "running"
              ? "进行中"
              : current.status === "done"
                ? "已完成"
                : current.status === "error"
                  ? "失败"
                  : "待命"}
          </span>
        </header>

        {mode === "chat" ? (
          <ChatPanel sessionId={sessionId} />
        ) : (
          <>
            <div className="stream">
              {notice && <p className="note error">{notice}</p>}
              {current.error && <p className="note error">{current.error}</p>}
              {!selected && current.plan.length === 0 && current.events.length === 0 && (
                <p className="note">
                  点击“开始诊断”，Agent 会先列出服务，再按症状选择指标、日志和进程查询；
                  失败的一步会被 verifier 拦下并触发重新规划。
                </p>
              )}

              {current.plan.length > 0 && (
                <>
                  <InspectorHeading text="执行计划" />
                  <PlanSpine live={current} />
                </>
              )}

              {current.events.some(
                (e) =>
                  e.event === "tool_step" ||
                  e.event === "decision" ||
                  e.event === "planner_route",
              ) && (
                <>
                  <InspectorHeading text="执行记录" />
                  <EventLog events={current.events} />
                </>
              )}

              {current.report && (
                <>
                  <InspectorHeading text="诊断报告" />
                  <article
                    className="report"
                    dangerouslySetInnerHTML={{ __html: renderMarkdown(current.report) }}
                  />
                </>
              )}
              <div ref={streamEnd} />
            </div>

            <div className="compose">
              <label htmlFor="session">会话 ID（同一 ID 的多次诊断不会累计历史步骤）</label>
              <div className="compose-row">
                <textarea
                  id="session"
                  rows={1}
                  value={sessionId}
                  onChange={(event) => setSessionId(event.target.value.trim())}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      void start();
                    }
                  }}
                />
                <button
                  className="btn"
                  onClick={() => void start()}
                  disabled={live.status === "running"}
                >
                  运行
                </button>
              </div>
            </div>
          </>
        )}
      </main>

      <aside className="inspector">
        <section>
          <h3>本次运行</h3>
          <dl className="kv">
            <dt>planner</dt>
            <dd>
              {current.backend ? (
                <span className={`badge ${current.backend}`}>
                  {current.backend === "local" ? "本地小模型" : "云端"}
                </span>
              ) : (
                "—"
              )}
            </dd>
            <dt>步骤</dt>
            <dd>
              {current.completed.size}/{current.plan.length}
            </dd>
            <dt>事件</dt>
            <dd>{current.events.length}</dd>
            <dt>tokens</dt>
            <dd>{sumTokens(current.events)}</dd>
          </dl>
        </section>

        <section>
          <h3>评测结果</h3>
          {evals.length === 0 ? (
            <p className="empty">还没有评测输出。运行 `python scripts/run_eval.py --live`。</p>
          ) : (
            evals.slice(0, 4).map((result) => (
              <div key={result.name} style={{ marginBottom: 14 }}>
                <div className="sub" style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)" }}>
                  {result.name} · {result.rows.length} cases
                </div>
                <table className="eval-table">
                  <thead>
                    <tr>
                      <th>指标</th>
                      <th>值</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>tool F1</td>
                      <td className="num">{result.mean_tool_f1.toFixed(3)}</td>
                    </tr>
                    <tr>
                      <td>任务成功率</td>
                      <td className="num">{(result.success_rate * 100).toFixed(0)}%</td>
                    </tr>
                    <tr>
                      <td>总 tokens</td>
                      <td className="num">{result.total_tokens.toLocaleString()}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            ))
          )}
        </section>

        <section>
          <h3>这套系统在做什么</h3>
          <p className="note" style={{ fontSize: 12.5 }}>
            Planner 输出带工具名的结构化计划，执行器直接调用 MCP 工具；verifier 只做确定性检查，
            失败时 replan 重写剩余步骤，最多两次。每次运行写入
            <code style={{ fontFamily: "var(--mono)" }}> logs/traces/&lt;run_id&gt;.jsonl</code>。
          </p>
        </section>
      </aside>
    </div>
  );
}

function InspectorHeading({ text }: { text: string }) {
  return <h3 style={{ margin: "14px 0 8px", fontSize: 11.5, fontWeight: 500, color: "var(--muted)" }}>{text}</h3>;
}

function applyEvent(prev: Live, event: RunEvent): Live {
  const events = [...prev.events, event];
  const next: Live = { ...prev, events };

  if (event.event === "planner_route") {
    next.backend = event.backend === "local" ? "local" : "cloud";
  }
  if (event.event === "plan" && Array.isArray(event.steps)) {
    next.plan = event.steps as PlanStep[];
  }
  if ((event.event === "tool_step" || event.event === "step_complete") && event.step) {
    const completed = new Set(prev.completed);
    completed.add(event.step);
    next.completed = completed;
  }
  if (event.event === "replan" && Array.isArray(event.steps)) {
    const replaced = { ...prev.replaced };
    const pending = prev.plan.filter((step) => !prev.completed.has(step.description ?? ""));
    if (pending.length) {
      replaced[pending[0].description ?? ""] = event.steps as PlanStep[];
    }
    next.plan = [
      ...prev.plan.filter((step) => prev.completed.has(step.description ?? "")),
      ...(event.steps as PlanStep[]),
    ];
    next.replaced = replaced;
  }
  if (event.event === "final_response") {
    const preview = String(event.result_preview ?? "");
    if (preview) next.report = preview;
  }
  if (event.event === "run_complete" && event.result_preview) {
    next.report = String(event.result_preview);
  }
  return next;
}

function replay(sessionId: string, events: RunEvent[]): Live {
  let live = emptyLive(sessionId);
  for (const event of events) {
    live = applyEvent(live, event);
  }
  if (events.some((e) => e.event === "run_complete")) live.status = "done";
  if (events.some((e) => e.event === "run_error")) live.status = "error";
  return live;
}

function sumTokens(events: RunEvent[]) {
  const total = events.reduce(
    (sum, event) => sum + (event.input_tokens ?? 0) + (event.output_tokens ?? 0),
    0,
  );
  return total.toLocaleString();
}

function PlanSpine({ live }: { live: Live }) {
  return (
    <ol className="plan">
      {live.plan.map((step, index) => {
        const text = step.description ?? `步骤 ${index + 1}`;
        const done = live.completed.has(text);
        const active = !done && live.status === "running" && index === live.completed.size;
        const replaced = live.replaced[text];
        return (
          <li
            key={`${step.id ?? index}-${text}`}
            className={`plan-step${done ? " done" : ""}${active ? " active" : ""}`}
          >
            <span className="step-text">{text}</span>
            {step.tool && <span className="tag">{step.tool}</span>}
            {replaced?.map((replacement, i) => (
              <span key={i} className="replaced">
                ↳ {replacement.description ?? replacement.tool}
              </span>
            ))}
          </li>
        );
      })}
    </ol>
  );
}

function EventLog({ events }: { events: RunEvent[] }) {
  const rows = events.filter((event) =>
    ["planner_route", "tool_step", "decision", "replan", "run_complete", "run_error"].includes(
      event.event,
    ),
  );
  return (
    <div className="log">
      {rows.map((event, index) => {
        const name = event.tools?.map((tool) => tool.name).join(", ") ?? "";
        let value = "";
        let kind = "log-row";
        if (event.event === "planner_route") {
          value = event.backend === "local" ? "本地 planner 命中" : `回退云端（${event.reason ?? "—"}）`;
          kind += " route";
        } else if (event.event === "tool_step") {
          value = `${name}${event.result_chars ? ` · ${event.result_chars} chars` : ""}`;
        } else if (event.event === "decision") {
          value = `decision: ${event.action ?? "—"}`;
        } else if (event.event === "replan") {
          value = `replan: ${event.reason ?? "重写剩余计划"}`;
          kind += " alert";
        } else if (event.event === "run_error") {
          value = `失败：${(event as unknown as Record<string, string>).message ?? ""}`;
          kind += " alert";
        } else if (event.event === "run_complete") {
          value = `完成 · ${fmtDuration(event.duration_ms)}`;
        }
        return (
          <div key={index} className={kind}>
            <span className="t">{fmtTime(event.ts)}</span>
            <span className="k">{event.event}</span>
            <span className="v">{value}</span>
          </div>
        );
      })}
    </div>
  );
}
