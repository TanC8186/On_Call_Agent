# web

On-Call Agent 的运行面板：查看每次诊断的计划、工具调用、replan 决策与 token 成本。

## 开发

```bash
npm install
npm run dev      # http://localhost:5173，/api 代理到 :9900
```

后端先起：

```bash
.venv/Scripts/python.exe -m uvicorn app.main:app --port 9900
```

## 构建

```bash
npm run build    # 产物在 dist/，由 FastAPI 直接托管
```

打开 `http://127.0.0.1:9900`。

## 结构

| 文件 | 作用 |
|---|---|
| `src/App.tsx` | 三栏布局：运行列表 / 计划脊线 + 日志 + 报告 / 指标检查器 |
| `src/api.ts` | REST 调用 + SSE 解析（`parseSseChunk`、`normalizeSseEvent`） |
| `src/markdown.ts` | 报告渲染，HTML 标签转义 |
| `src/styles.css` | 全部样式；IBM Plex Sans/Mono，无 UI 框架依赖 |

## 约定

- 后端 SSE 用 `type` 字段，trace 文件用 `event` 字段；`normalizeSseEvent` 统一成 trace 形状，组件只处理一种结构。
- SSE 由 `sse-starlette` 发出，行结束符是 CRLF，解析前必须先归一化。
- 环境变量：`VITE_API_BASE`（跨域部署时用）、`VITE_API_KEY`（后端设了 `API_KEY` 时）。
