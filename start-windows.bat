@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ====================================
echo 启动 OnCallAgent 服务
echo ====================================
echo.

REM 检查 uv 是否安装（可选，如果没有会使用 pip）
echo [1/7] 检查包管理器...
where uv >nul 2>&1
if errorlevel 1 (
    echo [信息] uv 未安装，将使用传统 pip 方式
    echo [提示] 安装 uv 可提升速度：pip install uv
    set USE_UV=0
) else (
    echo [成功] 检测到 uv 包管理器
    set USE_UV=1
)
echo.

REM 确保 Python 版本正确
echo [2/7] 配置 Python 环境...
if exist .venv\Scripts\python.exe (
    echo [信息] 虚拟环境已存在
) else (
    echo [信息] 创建新的虚拟环境...
    python -m venv .venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败
        echo [提示] 请确保已安装 Python 3.11+
        pause
        exit /b 1
    )
    .venv\Scripts\python.exe -m pip install --upgrade pip -q
    .venv\Scripts\python.exe -m pip install -e . -q
)
echo [成功] 虚拟环境就绪
echo.

REM 设置 Python 命令
set PYTHON_CMD=.venv\Scripts\python.exe

REM 启动 Docker Compose
echo [3/7] 启动 Milvus 向量数据库...
docker ps --format "{{.Names}}" | findstr "milvus-standalone" >nul 2>&1
if not errorlevel 1 (
    echo [信息] Milvus 容器已在运行
) else (
    docker compose -f vector-database.yml up -d
    if errorlevel 1 (
        echo [错误] Docker 启动失败，请确保 Docker Desktop 已启动
        pause
        exit /b 1
    )
    echo [信息] 等待 Milvus 启动（10秒）...
    timeout /t 10 /nobreak >nul
)
echo [成功] Milvus 数据库就绪
echo.

REM 启动 CLS MCP 服务
echo [4/7] 启动 CLS MCP 服务（端口 8003）...
start "CLS MCP Server" /min %PYTHON_CMD% servers/cls_server.py
timeout /t 2 /nobreak >nul
echo [成功] CLS MCP 服务已启动
echo.

REM 启动 Monitor MCP 服务
echo [5/7] 启动 Monitor MCP 服务（端口 8004）...
start "Monitor MCP Server" /min %PYTHON_CMD% servers/monitor_server.py
timeout /t 2 /nobreak >nul
echo [成功] Monitor MCP 服务已启动
echo.

REM 启动 FastAPI Web 服务
echo [6/7] 启动 FastAPI Web 服务（端口 9900）...
start "OnCallAgent API" %PYTHON_CMD% servers/web_server.py
echo [信息] 等待服务启动（15秒）...
timeout /t 15 /nobreak >nul
echo.

REM 检查服务状态
echo [7/7] 检查服务状态...
curl -s http://localhost:9900/health >nul 2>&1
if errorlevel 1 (
    echo [警告] 服务可能还未完全启动，请稍等片刻
) else (
    echo [成功] FastAPI 服务运行正常
)

echo.
echo ====================================
echo 服务启动完成！
echo ====================================
echo Web 界面: http://localhost:9900
echo API 文档: http://localhost:9900/docs
echo.
echo 所有启动脚本已统一到 servers/ 目录:
echo   - servers/cls_server.py      (端口 8003)
echo   - servers/monitor_server.py  (端口 8004)
echo   - servers/web_server.py      (端口 9900)
echo.
echo 停止服务: stop-windows.bat
echo ====================================
pause
