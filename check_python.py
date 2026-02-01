#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python版本检查脚本
用于检查当前Python版本是否兼容PyCaret库
"""

import sys
import platform
import subprocess
import os

def check_python_version():
    """检查Python版本是否兼容"""
    python_version = sys.version_info
    version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
    
    print(f"\n当前Python版本: {version_str}")
    print(f"系统平台: {platform.system()} {platform.release()}")
    
    # 检查版本兼容性
    if python_version.major == 3 and (python_version.minor == 10 or python_version.minor == 11):
        print("✓ Python版本兼容 - 可以正常使用PyCaret")
        compatible = True
    else:
        print("✗ Python版本不兼容 - PyCaret推荐使用Python 3.10或3.11")
        compatible = False
    
    return compatible

def check_pip_packages():
    """检查是否已安装关键包"""
    try:
        # 检查pip是否可用
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                       check=True, capture_output=True, text=True)
        
        # 检查venv模块
        try:
            import venv
            print("✓ venv模块已安装 - 可以创建虚拟环境")
        except ImportError:
            print("✗ venv模块未安装 - 请安装venv以创建虚拟环境")
            return False
        
        return True
    except subprocess.CalledProcessError:
        print("✗ pip未安装或不可用 - 请确保pip已正确安装")
        return False

def main():
    print("===== Python环境检查 =====")
    
    # 检查Python版本
    version_ok = check_python_version()
    
    # 检查pip和venv
    pip_ok = check_pip_packages()
    
    print("\n===== 检查结果 =====")
    if version_ok and pip_ok:
        print("✓ 您的Python环境适合运行PyCaret项目")
        print("  请运行setup_env.bat创建虚拟环境并安装依赖")
    else:
        print("✗ 您的Python环境需要调整才能运行PyCaret项目")
        if not version_ok:
            print("  建议安装Python 3.10或3.11版本")
            print("  下载地址: https://www.python.org/downloads/")
        if not pip_ok:
            print("  请确保pip和venv模块已正确安装")
    
    print("\n按Enter键退出...")
    input()

if __name__ == "__main__":
    main()