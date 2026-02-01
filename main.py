# 主程序文件
# 用于调度整个项目的工作流程

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from src.utils.config_loader import create_config_loader
from src.preprocessing import DataPreprocessor
from src.feature_selection import FeatureSelector
from src.train_model import ModelTrainer
from src.evaluate_model import ModelEvaluator


def setup_directories(config_loader):
    """
    设置项目目录
    
    Args:
        config_loader: 配置加载器实例
    """
    # 获取路径配置
    paths_config = config_loader.get_paths_config()
    
    # 创建目录
    os.makedirs(paths_config['data']['raw_dir'], exist_ok=True)
    os.makedirs(paths_config['data']['cleaned_dir'], exist_ok=True)
    os.makedirs(paths_config['data']['selected_features_dir'], exist_ok=True)
    os.makedirs(paths_config['output']['models_dir'], exist_ok=True)
    os.makedirs(paths_config['output']['figures_dir'], exist_ok=True)
    os.makedirs(paths_config['output']['results_dir'], exist_ok=True)


def copy_raw_data(config_loader, source_dir):
    """
    复制原始数据到项目目录
    
    Args:
        config_loader: 配置加载器实例
        source_dir: 原始数据源目录
    """
    # 获取路径配置
    paths_config = config_loader.get_paths_config()
    
    # 源文件路径
    covariates_source = os.path.join(source_dir, paths_config['data']['covariates_file'])
    outcomes_source = os.path.join(source_dir, paths_config['data']['outcomes_file'])
    
    # 目标文件路径
    covariates_target = os.path.join(paths_config['data']['raw_dir'], paths_config['data']['covariates_file'])
    outcomes_target = os.path.join(paths_config['data']['raw_dir'], paths_config['data']['outcomes_file'])
    
    # 复制文件
    if os.path.exists(covariates_source):
        shutil.copy2(covariates_source, covariates_target)
        print(f"已复制协变量文件: {covariates_source} -> {covariates_target}")
    else:
        print(f"警告: 协变量文件 {covariates_source} 不存在")
    
    if os.path.exists(outcomes_source):
        shutil.copy2(outcomes_source, outcomes_target)
        print(f"已复制结局变量文件: {outcomes_source} -> {outcomes_target}")
    else:
        print(f"警告: 结局变量文件 {outcomes_source} 不存在")


def run_pipeline(config_loader, steps=None):
    """
    运行完整的数据处理和模型训练流程
    
    Args:
        config_loader: 配置加载器实例
        steps: 要执行的步骤列表，如果为None则执行所有步骤
    """
    # 如果steps为None，则执行所有步骤
    if steps is None:
        steps = ['preprocess', 'select_features', 'train_models', 'evaluate_models']
    
    # 数据预处理
    if 'preprocess' in steps:
        print("\n===== 步骤1: 数据预处理 =====")
        preprocessor = DataPreprocessor(config_loader)
        df = preprocessor.preprocess()
        print("数据预处理完成\n")
    
    # 特征选择
    if 'select_features' in steps:
        print("\n===== 步骤2: 特征选择 =====")
        selector = FeatureSelector(config_loader)
        final_df = selector.select_features()
        print("特征选择完成\n")
    
    # 模型训练
    if 'train_models' in steps:
        print("\n===== 步骤3: 模型训练 =====")
        trainer = ModelTrainer(config_loader)
        models = trainer.train_all_models()
        print("模型训练完成\n")
    
    # 模型评估
    if 'evaluate_models' in steps:
        print("\n===== 步骤4: 模型评估 =====")
        evaluator = ModelEvaluator(config_loader)
        evaluator.evaluate_models()
        print("模型评估完成\n")


def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='FI预测CAD项目')
    parser.add_argument('--source_dir', type=str, default='d:\\Workspace\\Testing\\Python\\FI预测 CAD\\思路1',
                        help='原始数据源目录')
    parser.add_argument('--steps', type=str, nargs='+',
                        choices=['preprocess', 'select_features', 'train_models', 'evaluate_models'],
                        help='要执行的步骤')
    args = parser.parse_args()
    
    # 创建配置加载器
    config_loader = create_config_loader()
    
    # 设置项目目录
    setup_directories(config_loader)
    
    # 复制原始数据
    copy_raw_data(config_loader, args.source_dir)
    
    # 运行流程
    run_pipeline(config_loader, args.steps)
    
    print("\n===== 项目执行完成 =====")


if __name__ == "__main__":
    main()