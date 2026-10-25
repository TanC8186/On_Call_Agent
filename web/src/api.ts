export type RunStatus = "ok" | "error";

export interface RunSummary {
  run_id: string;
  session_id: string | null;
  started_at: string | null;
  duration_ms: number | null;
  status: RunStatus;
  steps: number;
  plan_steps: number;
  backend: "local" | "cloud";
  severity: string;
  category: string;
  input_tokens: number;
  output_tokens: number;
}

export interface RunEvent {
  ts: string;
  event: string;
  node?: string;
  step?: string;
  tools?: { name: string; args: Record<string, unknown> }[];
  action?: string;
  backend?: string;
  reason?: string;
  severity?: string;
  category?: string;
  steps?: unknown;
  input_tokens?: number | null;
  output_tokens?: number | null;
  result_chars?: number;
  result_preview?: string;
  duration_ms?: number;
  message?: string;
  run_id?: string;
  session_id?: string;
}

export interface EvalRow {
  id: string;
  tool_f1?: number;
  tool_recall?: number;
  tool_precision?: number;
  task_success?: boolean;
  steps?: number;
  input_tokens?: number;
  output_tokens?: number;
  groundedness?: number;
  facts_matched?: number;
  facts_total?: number;
  error?: string;
}

export interface EvalResult {
  name: string;
  rows: EvalRow[];
  mean_tool_f1: number;
  success_rate: number;
  total_tokens: number;
}

export const API_BASE = import.meta.env.VITE_API_BASE ?? "";
const API_KEY = import.meta.env.VITE_API_KEY ?? "";

export function headers(): Record<string, string> {
  const result: Record<string, string> = { "Content-Type": "application/json" };
  if (API_KEY) result["X-API-Key"] = API_KEY;
  return result;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { headers: headers() });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export const api = {
  listRuns: () => getJson<{ runs: RunSummary[] }>("/api/runs"),
  getRun: (runId: string) =>
    getJson<{ run_id: string; events: RunEvent[] }>(`/api/runs/${runId}`),
  listEvalResults: () => getJson<{ results: EvalResult[] }>("/api/eval/results"),
  health: () => getJson<Record<string, unknown>>("/health"),
};

export interface DiagnoseHandlers {
  onEvent: (event: RunEvent) => void;
  onDone: () => void;
  onError: (message: string) => void;
}

const CRLF = String.fromCharCode(13) + String.fromCharCode(10);
const LF = String.fromCharCode(10);

/** 从一段 SSE 原始文本里解析出 data: JSON 事件。 */
export function parseSseChunk(chunk: string): RunEvent[] {
  const events: RunEvent[] = [];
  const dataLines = chunk
    .split(LF)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim());
  if (!dataLines.length) return events;
  const raw = dataLines.join(LF);
  if (!raw) return events;
  try {
    events.push(normalizeSseEvent(JSON.parse(raw) as Record<string, unknown>));
  } catch {
    // 忽略心跳或非 JSON 负载
  }
  return events;
}

/**
 * 后端 SSE 用 `type` 字段，trace 文件用 `event` 字段。
 * 统一成 trace 的形状，前端只处理一种结构。
 */
export function normalizeSseEvent(payload: Record<string, unknown>): RunEvent {
  const type = String(payload.type ?? payload.event ?? "unknown");
  const base: RunEvent = {
    ts: String(payload.ts ?? new Date().toISOString()),
    event: type,
  };
  if (type === "plan") {
    base.steps = payload.plan_steps ?? payload.plan;
  }
  if (type === "step_complete") {
    base.step = payload.current_step as string | undefined;
    base.result_chars = (payload.result_preview as string | undefined)?.length;
  }
  if (type === "plan") {
    base.steps = payload.plan_steps ?? payload.plan;
  }
  if (type === "report") {
    base.result_preview = payload.report as string | undefined;
  }
  if (type === "complete") {
    base.result_preview = payload.response as string | undefined;
    base.duration_ms = payload.duration_ms as number | undefined;
  }
  return { ...base, ...(payload as unknown as Partial<RunEvent>), event: type };
}

/**
 * 用 fetch + ReadableStream 消费 SSE：EventSource 不支持 POST。
 * sse-starlette 用 CRLF 分隔行，所以先归一化再按空行切事件。
 */
export async function diagnose(
  sessionId: string,
  handlers: DiagnoseHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/aiops`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ session_id: sessionId }),
    signal,
  });
  if (!response.ok || !response.body) {
    handlers.onError(`诊断请求失败: ${response.status}`);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).split(CRLF).join(LF);

      const chunks = buffer.split(LF + LF);
      buffer = chunks.pop() ?? "";
      for (const chunk of chunks) {
        for (const event of parseSseChunk(chunk)) {
          handlers.onEvent(event);
          if (event.event === "complete") {
            handlers.onDone();
            return;
          }
          if (event.event === "error") {
            handlers.onError(event.message ?? "诊断失败");
            return;
          }
        }
      }
    }
    handlers.onDone();
  } catch (error) {
    if ((error as Error).name !== "AbortError") {
      handlers.onError((error as Error).message);
    }
  }
}
