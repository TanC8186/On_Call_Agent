/** 对话（RAG）接口：与 AIOps 诊断共用同一套 SSE 解析。 */

import { API_BASE, headers, normalizeSseEvent, type RunEvent } from "./api";

export interface ChatHandlers {
  onDelta: (text: string) => void;
  onEvent?: (event: RunEvent) => void;
  onDone: () => void;
  onError: (message: string) => void;
}

const CRLF = String.fromCharCode(13) + String.fromCharCode(10);
const LF = String.fromCharCode(10);

/**
 * POST /api/chat_stream。
 * 后端事件：content / tool_call / search_results / done / error。
 */
export async function chatStream(
  sessionId: string,
  question: string,
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/chat_stream`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ Id: sessionId, Question: question }),
    signal,
  });
  if (!response.ok || !response.body) {
    handlers.onError(`对话请求失败: ${response.status}`);
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
        for (const line of chunk.split(LF)) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          if (!raw) continue;
          let payload: Record<string, unknown>;
          try {
            payload = JSON.parse(raw) as Record<string, unknown>;
          } catch {
            continue;
          }
          const type = String(payload.type ?? "");
          if (type === "content") {
            handlers.onDelta(String(payload.data ?? ""));
          } else if (type === "tool_call") {
            handlers.onEvent?.({
              ts: new Date().toISOString(),
              event: "tool_call",
              message: JSON.stringify(payload.data),
            });
          } else if (type === "done") {
            handlers.onDone();
            return;
          } else if (type === "error") {
            handlers.onError(String(payload.data ?? "对话失败"));
            return;
          } else if (handlers.onEvent) {
            handlers.onEvent(normalizeSseEvent(payload));
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
