@echo off
echo ===== 运行FI预测CVD项目 =====
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
echo 环境已激活，正在运行项目...
echo.

REM 运行主程序
python main.py

echo.
echo 程序执行完毕
echo.

pause