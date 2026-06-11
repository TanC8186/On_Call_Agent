#!/bin/bash
set -e

echo "============================================"
echo "  On-Call Agent — Docker 容器启动"
echo "============================================"

# 等待 Milvus 就绪
echo "[1/4] 等待 Milvus 启动..."
for i in $(seq 1 30); do
    if curl -s "http://${MILVUS_HOST:-milvus-standalone}:9091/healthz" > /dev/null 2>&1; then
        echo "  ✅ Milvus 已就绪"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "  ⚠️  Milvus 连接超时，RAG 功能将不可用"
    fi
    sleep 2
done

# 启动 MCP 服务
echo "[2/4] 启动 MCP 服务..."
python servers/cls_server.py &
PID_CLS=$!
echo "  CLS MCP Server    → :8003 (PID $PID_CLS)"

python servers/monitor_server.py &
PID_MONITOR=$!
echo "  Monitor MCP Server → :8004 (PID $PID_MONITOR)"

python servers/ssh_server.py &
PID_SSH=$!
echo "  SSH MCP Server    → :8005 (PID $PID_SSH)"

# 等待 MCP 服务就绪
sleep 3
echo "[3/4] MCP 服务已全部启动"

# 启动 Web 服务（前台运行，保持容器存活）
echo "[4/4] 启动 Web 服务 → :9900"
echo "============================================"
echo "  🌐 访问: http://localhost:9900"
echo "  📚 API:  http://localhost:9900/docs"
echo "============================================"

exec python servers/web_server.py
