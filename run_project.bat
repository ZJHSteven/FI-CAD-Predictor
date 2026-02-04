@echo off
setlocal

echo ===== 运行FI预测CVD项目 (UV) =====
echo.

REM 检查uv是否可用
uv --version
IF %ERRORLEVEL% NEQ 0 (
    echo 未检测到uv，请先安装uv并确保加入PATH
    pause
    exit /b 1
)

echo 正在运行主流程...
uv run python main.py

echo.
echo 程序执行完毕
echo.
pause
endlocal
