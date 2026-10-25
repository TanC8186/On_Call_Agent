import { Marked } from "marked";

const marked = new Marked({
  gfm: true,
  breaks: true,
});

marked.use({
  renderer: {
    html(token) {
      // 报告来自本地 LLM/MCP，仍按纯文本处理，避免注入
      return token.text.replace(/</g, "&lt;");
    },
  },
});

export function renderMarkdown(source: string): string {
  return marked.parse(source, { async: false }) as string;
}
