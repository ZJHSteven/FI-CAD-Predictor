@echo off
setlocal

echo ===== 启动FI/CVD预测API =====
echo.

REM 检查虚拟环境是否存在
IF NOT EXIST venv\Scripts\activate.bat (
    echo 虚拟环境未创建，请先运行 setup_env.bat
    pause
    exit /b 1
)

REM 激活虚拟环境
echo 正在激活虚拟环境...
call venv\Scripts\activate.bat

echo.
echo 环境已激活，正在启动API服务...
echo.

REM 启动FastAPI (支持热更新)
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

echo.
echo API服务已退出
echo.

pause
endlocal
