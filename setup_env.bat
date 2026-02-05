@echo off
setlocal
REM 解决中文乱码：切换为GBK代码页
chcp 936 >nul

echo ===== 使用UV准备推理环境 =====
echo.

REM 检查uv是否可用
uv --version
IF %ERRORLEVEL% NEQ 0 (
    echo 未检测到uv，请先安装uv并确保加入PATH
    pause
    exit /b 1
)

REM 安装并固定Python 3.10（与PyCaret兼容）
echo 正在安装Python 3.10...
uv python install 3.10
IF %ERRORLEVEL% NEQ 0 (
    echo Python安装失败，请检查网络或权限
    pause
    exit /b 1
)

REM 同步依赖并创建.venv
echo 正在同步依赖并创建环境...
uv sync
IF %ERRORLEVEL% NEQ 0 (
    echo 依赖安装失败，请检查pyproject.toml与uv.lock
    pause
    exit /b 1
)

echo.
echo ===== 环境设置完成 =====
echo 使用说明：
echo 1. 启动API：uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
echo 2. 生产部署建议：使用systemd或Docker管理进程，并避免--reload
echo.
pause
endlocal
