import { useCallback, useEffect, useRef, useState } from "react";
import { chatStream } from "./chat";
import { renderMarkdown } from "./markdown";

interface Turn {
  id: number;
  role: "user" | "assistant";
  text: string;
  streaming?: boolean;
  toolCalls?: string[];
}

interface Props {
  sessionId: string;
}

/** RAG 对话面板：与诊断共用 session，回答走 /api/chat_stream。 */
export default function ChatPanel({ sessionId }: Props) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const nextId = useRef(1);
  const abortRef = useRef<AbortController | null>(null);
  const listEnd = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    listEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length, busy]);

  const send = useCallback(async () => {
    const question = draft.trim();
    if (!question || busy) return;

    const userTurn: Turn = { id: nextId.current++, role: "user", text: question };
    const assistantId = nextId.current++;
    setTurns((prev) => [
      ...prev,
      userTurn,
      { id: assistantId, role: "assistant", text: "", streaming: true, toolCalls: [] },
    ]);
    setDraft("");
    setBusy(true);
    setError("");

    const controller = new AbortController();
    abortRef.current = controller;

    const patch = (update: (turn: Turn) => Turn) => {
      setTurns((prev) =>
        prev.map((turn) => (turn.id === assistantId ? update(turn) : turn)),
      );
    };

    await chatStream(
      sessionId,
      question,
      {
        onDelta: (text) => patch((turn) => ({ ...turn, text: turn.text + text })),
        onEvent: (event) => {
          if (event.event !== "tool_call") return;
          const label = event.message ?? "";
          patch((turn) => ({
            ...turn,
            toolCalls: [...(turn.toolCalls ?? []), label],
          }));
        },
        onDone: () => {
          patch((turn) => ({ ...turn, streaming: false }));
          setBusy(false);
        },
        onError: (message) => {
          patch((turn) => ({ ...turn, streaming: false }));
          setError(message);
          setBusy(false);
        },
      },
      controller.signal,
    );
  }, [draft, busy, sessionId]);

  return (
    <div className="chat">
      <div className="chat-log">
        {turns.length === 0 && (
          <p className="note">
            基于运维知识库的问答。上传的 runbook 会被切分、向量化，回答时检索 top-k 片段；
            涉及实时状态的问题会自动调用 MCP 工具。
          </p>
        )}
        {turns.map((turn) => (
          <article key={turn.id} className={`turn ${turn.role}`}>
            <span className="turn-role">{turn.role === "user" ? "你" : "Agent"}</span>
            {turn.role === "assistant" ? (
              <div
                className="turn-body"
                dangerouslySetInnerHTML={{ __html: renderMarkdown(turn.text) }}
              />
            ) : (
              <p className="turn-body">{turn.text}</p>
            )}
            {turn.streaming && <span className="caret" aria-label="生成中" />}
            {!!turn.toolCalls?.length && (
              <ul className="tool-trail">
                {turn.toolCalls.map((call, index) => (
                  <li key={index}>{call}</li>
                ))}
              </ul>
            )}
          </article>
        ))}
        {error && <p className="note error">{error}</p>}
        <div ref={listEnd} />
      </div>

      <div className="compose">
        <label htmlFor="question">问题</label>
        <div className="compose-row">
          <textarea
            id="question"
            rows={2}
            value={draft}
            placeholder="例如：磁盘满了应该先做哪几步？"
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void send();
              }
            }}
          />
          <button className="btn" onClick={() => void send()} disabled={busy || !draft.trim()}>
            {busy ? "回答中…" : "发送"}
          </button>
          {busy && (
            <button
              className="btn ghost"
              onClick={() => {
                abortRef.current?.abort();
                setBusy(false);
              }}
            >
              停止
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
