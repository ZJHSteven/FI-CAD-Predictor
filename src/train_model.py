# 模型训练模块
# 用于训练不同类型的模型

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pickle

# 模型导入
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score

from src.utils.config_loader import create_config_loader


class ModelTrainer:
    """
    模型训练类，用于训练不同类型的模型
    """
    
    def __init__(self, config_loader=None):
        """
        初始化模型训练器
        
        Args:
            config_loader: 配置加载器实例，如果为None则创建新实例
        """
        if config_loader is None:
            self.config_loader = create_config_loader()
        else:
            self.config_loader = config_loader
        
        # 加载配置
        self.paths_config = self.config_loader.get_paths_config()
        self.model_config = self.config_loader.get_model_config()
        self.variables_config = self.config_loader.get_variables_config()
        
        # 设置路径
        self.selected_features_dir = self.paths_config['data']['selected_features_dir']
        self.models_dir = self.paths_config['output']['models_dir']
        
        # 创建输出目录
        os.makedirs(self.models_dir, exist_ok=True)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        加载特征选择后的数据
        
        Returns:
            特征矩阵和目标变量
        """
        # 获取文件路径
        features_path = os.path.join(self.selected_features_dir, 'final_features.csv')
        
        # 检查文件是否存在
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"特征选择后的数据文件 {features_path} 不存在")
        
        # 读取数据
        df = pd.read_csv(features_path)
        
        # 创建目标变量
        y = df['CVD']
        
        # 获取特征矩阵（排除目标变量）
        X = df.drop(['CVD'], axis=1, errors='ignore')
        
        print(f"加载数据完成，特征矩阵形状: {X.shape}，目标变量形状: {y.shape}")
        return X, y
    
    def train_logistic_regression(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict[str, float]]:
        """
        训练Logistic回归模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            训练好的模型和评估指标
        """
        print("\n===== 训练Logistic回归模型 =====")
        
        # 获取模型配置
        logistic_config = self.model_config['logistic']
        
        # 拆分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=logistic_config['random_state']
        )
        
        # 创建模型
        model = LogisticRegression(
            C=logistic_config['C'],
            penalty=logistic_config['penalty'],
            max_iter=logistic_config['max_iter'],
            random_state=logistic_config['random_state']
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 评估模型
        metrics = self._evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # 如果使用交叉验证
        if logistic_config['use_cv']:
            cv_score = cross_val_score(
                model, X, y, 
                cv=logistic_config['cv_folds'], 
                scoring='roc_auc'
            ).mean()
            metrics['cv_auc'] = cv_score
            print(f"交叉验证AUC: {cv_score:.4f}")
        
        # 保存模型
        model_path = os.path.join(self.models_dir, self.paths_config['output']['logistic_model'])
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Logistic回归模型已保存到 {model_path}")
        
        return model, metrics
    
    def train_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict[str, float]]:
        """
        训练XGBoost模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            训练好的模型和评估指标
        """
        print("\n===== 训练XGBoost模型 =====")
        
        # 获取模型配置
        xgboost_config = self.model_config['xgboost']
        
        # 拆分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=xgboost_config['random_state']
        )
        
        # 创建DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # 设置参数
        params = {
            'objective': xgboost_config['objective'],
            'eval_metric': xgboost_config['eval_metric'],
            'eta': xgboost_config['eta'],
            'max_depth': xgboost_config['max_depth'],
            'subsample': xgboost_config['subsample'],
            'colsample_bytree': xgboost_config['colsample_bytree'],
            'lambda': xgboost_config['lambda'],
            'alpha': xgboost_config['alpha']
        }
        
        # 训练模型
        evallist = [(dtrain, 'train'), (dtest, 'eval')]
        model = xgb.train(
            params, dtrain, 
            num_boost_round=xgboost_config['num_boost_round'], 
            evals=evallist,
            early_stopping_rounds=xgboost_config['early_stopping_rounds'], 
            verbose_eval=False
        )
        
        # 评估模型
        y_train_pred = model.predict(dtrain)
        y_test_pred = model.predict(dtest)
        
        from sklearn.metrics import roc_auc_score, accuracy_score
        train_auc = roc_auc_score(y_train, y_train_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)
        train_acc = accuracy_score(y_train, y_train_pred > 0.5)
        test_acc = accuracy_score(y_test, y_test_pred > 0.5)
        
        metrics = {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_acc': train_acc,
            'test_acc': test_acc
        }
        
        print(f"训练集AUC: {train_auc:.4f}, 准确率: {train_acc:.4f}")
        print(f"测试集AUC: {test_auc:.4f}, 准确率: {test_acc:.4f}")
        
        # 保存模型
        model_path = os.path.join(self.models_dir, self.paths_config['output']['xgboost_model'])
        model.save_model(model_path)
        print(f"XGBoost模型已保存到 {model_path}")
        
        return model, metrics
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict[str, float]]:
        """
        训练随机森林模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            训练好的模型和评估指标
        """
        print("\n===== 训练随机森林模型 =====")
        
        # 获取模型配置
        rf_config = self.model_config['random_forest']
        
        # 拆分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rf_config['random_state']
        )
        
        # 创建模型
        model = RandomForestClassifier(
            n_estimators=rf_config['n_estimators'],
            max_depth=rf_config['max_depth'],
            min_samples_leaf=rf_config['min_samples_leaf'],
            min_samples_split=rf_config['min_samples_split'],
            max_features=rf_config['max_features'],
            random_state=rf_config['random_state']
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 评估模型
        metrics = self._evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # 保存模型
        model_path = os.path.join(self.models_dir, self.paths_config['output']['rf_model'])
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"随机森林模型已保存到 {model_path}")
        
        return model, metrics
    
    def train_mlp(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict[str, float]]:
        """
        训练多层感知机模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            训练好的模型和评估指标
        """
        print("\n===== 训练多层感知机模型 =====")
        
        # 获取模型配置
        mlp_config = self.model_config['mlp']
        
        # 拆分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=mlp_config['random_state']
        )
        
        # 创建模型
        model = MLPClassifier(
            hidden_layer_sizes=tuple(mlp_config['hidden_layer_sizes']),
            activation=mlp_config['activation'],
            solver=mlp_config['solver'],
            learning_rate=mlp_config['learning_rate'],
            learning_rate_init=mlp_config['learning_rate_init'],
            max_iter=mlp_config['max_iter'],
            random_state=mlp_config['random_state']
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 评估模型
        metrics = self._evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # 保存模型
        model_path = os.path.join(self.models_dir, self.paths_config['output']['mlp_model'])
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"多层感知机模型已保存到 {model_path}")
        
        return model, metrics
    
    def _evaluate_model(self, model, X_train, X_test, y_train, y_test) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            X_train: 训练集特征
            X_test: 测试集特征
            y_train: 训练集标签
            y_test: 测试集标签
            
        Returns:
            评估指标字典
        """
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        # 预测概率
        y_train_pred = model.predict_proba(X_train)[:, 1]
        y_test_pred = model.predict_proba(X_test)[:, 1]
        
        # 计算AUC
        train_auc = roc_auc_score(y_train, y_train_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)
        
        # 计算准确率
        train_acc = accuracy_score(y_train, y_train_pred > 0.5)
        test_acc = accuracy_score(y_test, y_test_pred > 0.5)
        
        metrics = {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_acc': train_acc,
            'test_acc': test_acc
        }
        
        print(f"训练集AUC: {train_auc:.4f}, 准确率: {train_acc:.4f}")
        print(f"测试集AUC: {test_auc:.4f}, 准确率: {test_acc:.4f}")
        
        return metrics
    
    def train_all_models(self) -> Dict[str, Tuple[Any, Dict[str, float]]]:
        """
        训练所有模型
        
        Returns:
            模型和评估指标的字典
        """
        print("开始训练模型...")
        
        # 加载数据
        X, y = self.load_data()
        
        # 训练不同模型
        models = {}
        
        # Logistic回归
        logistic_model, logistic_metrics = self.train_logistic_regression(X, y)
        models['logistic'] = (logistic_model, logistic_metrics)
        
        # XGBoost
        xgboost_model, xgboost_metrics = self.train_xgboost(X, y)
        models['xgboost'] = (xgboost_model, xgboost_metrics)
        
        # 随机森林
        rf_model, rf_metrics = self.train_random_forest(X, y)
        models['random_forest'] = (rf_model, rf_metrics)
        
        # 多层感知机
        mlp_model, mlp_metrics = self.train_mlp(X, y)
        models['mlp'] = (mlp_model, mlp_metrics)
        
        # 比较模型性能
        print("\n===== 模型性能比较 =====")
        performance_df = pd.DataFrame({
            '模型': ['Logistic回归', 'XGBoost', '随机森林', '多层感知机'],
            '训练集AUC': [
                logistic_metrics['train_auc'],
                xgboost_metrics['train_auc'],
                rf_metrics['train_auc'],
                mlp_metrics['train_auc']
            ],
            '测试集AUC': [
                logistic_metrics['test_auc'],
                xgboost_metrics['test_auc'],
                rf_metrics['test_auc'],
                mlp_metrics['test_auc']
            ],
            '训练集准确率': [
                logistic_metrics['train_acc'],
                xgboost_metrics['train_acc'],
                rf_metrics['train_acc'],
                mlp_metrics['train_acc']
            ],
            '测试集准确率': [
                logistic_metrics['test_acc'],
                xgboost_metrics['test_acc'],
                rf_metrics['test_acc'],
                mlp_metrics['test_acc']
            ]
        })
        
        print(performance_df)
        
        # 保存性能比较结果
        os.makedirs(self.paths_config['output']['results_dir'], exist_ok=True)
        performance_path = os.path.join(self.paths_config['output']['results_dir'], 'model_performance.csv')
        performance_df.to_csv(performance_path, index=False)
        print(f"模型性能比较结果已保存到 {performance_path}")
        
        return models


# 如果直接运行此脚本，则执行模型训练
if __name__ == "__main__":
    trainer = ModelTrainer()
    models = trainer.train_all_models()