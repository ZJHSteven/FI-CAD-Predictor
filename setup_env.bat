@echo off
echo ===== 创建Python虚拟环境 (Python 3.10/3.11) =====
echo.

REM 检查Python版本
python --version
IF %ERRORLEVEL% NEQ 0 (
    echo Python未安装或不在PATH中，请先安装Python 3.10或3.11
    pause
    exit /b 1
)

echo.
echo 正在创建虚拟环境...

REM 创建虚拟环境
python -m venv venv
IF %ERRORLEVEL% NEQ 0 (
    echo 创建虚拟环境失败！请确保已安装venv模块
    pause
    exit /b 1
)

echo 虚拟环境创建成功！
echo.

REM 激活虚拟环境并安装依赖
echo 正在激活虚拟环境并安装依赖...
call venv\Scripts\activate.bat

echo 正在升级pip...
python -m pip install --upgrade pip

echo 正在安装项目依赖...
pip install -r requirements.txt

IF %ERRORLEVEL% NEQ 0 (
    echo 安装依赖失败！请检查网络连接或requirements.txt文件
    pause
    exit /b 1
)

echo.
echo ===== 环境设置完成 =====
echo.
echo 使用说明：
echo 1. 每次使用前，请先运行 venv\Scripts\activate.bat 激活环境
echo 2. 运行项目：python main.py
echo 3. 完成后，输入 deactivate 退出虚拟环境
echo.
echo 当前环境已激活，可以开始使用了！
echo.

pause