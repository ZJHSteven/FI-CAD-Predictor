@echo off
setlocal
REM 解决中文乱码：切换为GBK代码页
chcp 936 >nul

echo ===== 启动FI/CVD预测API (UV) =====
echo.

REM 检查uv是否可用
uv --version
IF %ERRORLEVEL% NEQ 0 (
    echo 未检测到uv，请先安装uv并确保加入PATH
    pause
    exit /b 1
)

echo 正在启动API服务...
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

echo.
echo API服务已退出
echo.
pause
endlocal
