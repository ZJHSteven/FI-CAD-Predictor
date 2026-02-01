# AutoML模块
# 用于自动训练和评估多种机器学习模型，并进行超参数优化

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns

# 模型导入
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 修改导入语句，使用相对路径导入
from utils.config_loader import create_config_loader


class AutoML:
    """
    AutoML类，用于自动训练和评估多种机器学习模型，并进行超参数优化
    """
    
    def __init__(self, config_loader=None):
        """
        初始化AutoML
        
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
        self.raw_dir = self.paths_config['data']['raw_dir']
        self.models_dir = self.paths_config['output']['models_dir']
        self.results_dir = self.paths_config['output']['results_dir']
        self.figures_dir = self.paths_config['output']['figures_dir']
        
        # 创建输出目录
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # 设置模型列表
        self.models = {
            'logistic': {
                'name': 'Logistic回归',
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'params': {
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                }
            },
            'random_forest': {
                'name': '随机森林',
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'xgboost': {
                'name': 'XGBoost',
                'model': xgb.XGBClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                }
            },
            'gradient_boosting': {
                'name': '梯度提升',
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0]
                }
            },
            'svm': {
                'name': '支持向量机',
                'model': Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, random_state=42))]),
                'params': {
                    'svm__C': [0.1, 1.0, 10.0],
                    'svm__kernel': ['linear', 'rbf', 'poly'],
                    'svm__gamma': ['scale', 'auto', 0.1, 0.01]
                }
            },
            'mlp': {
                'name': '多层感知机',
                'model': Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(max_iter=1000, random_state=42))]),
                'params': {
                    'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'mlp__activation': ['relu', 'tanh'],
                    'mlp__alpha': [0.0001, 0.001, 0.01],
                    'mlp__learning_rate': ['constant', 'adaptive']
                }
            },
            'knn': {
                'name': 'K近邻',
                'model': Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]),
                'params': {
                    'knn__n_neighbors': [3, 5, 7, 9, 11],
                    'knn__weights': ['uniform', 'distance'],
                    'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }
            }
        }
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        从原始数据文件加载数据
        
        Returns:
            特征矩阵和目标变量
        """
        # 获取文件路径
        outcomes_file = os.path.join(self.raw_dir, self.paths_config['data']['outcomes_file'])
        
        # 检查文件是否存在
        if not os.path.exists(outcomes_file):
            raise FileNotFoundError(f"原始数据文件 {outcomes_file} 不存在")
        
        # 读取数据
        df = pd.read_csv(outcomes_file)
        
        # 创建目标变量 (CVD)
        y = df['CVD'].astype(int)  # 将布尔值转换为整数
        
        # 获取特征矩阵（排除ID和目标变量）
        X = df.drop(['ID', 'CVD'], axis=1, errors='ignore')
        
        print(f"加载数据完成，特征矩阵形状: {X.shape}，目标变量形状: {y.shape}")
        print(f"特征列: {X.columns.tolist()}")
        
        return X, y
    
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5, search_type: str = 'random', n_iter: int = 20) -> Dict[str, Any]:
        """
        训练和评估多个模型，并进行超参数优化
        
        Args:
            X: 特征矩阵
            y: 目标变量
            cv_folds: 交叉验证折数
            search_type: 超参数搜索类型，'grid'或'random'
            n_iter: 随机搜索迭代次数
            
        Returns:
            模型评估结果字典
        """
        print(f"\n===== 开始AutoML模型训练和评估 (交叉验证折数: {cv_folds}) =====\n")
        
        # 拆分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = {}
        best_models = {}
        all_metrics = []
        
        # 遍历所有模型
        for model_key, model_info in self.models.items():
            print(f"\n----- 训练和评估 {model_info['name']} 模型 -----")
            start_time = time.time()
            
            # 创建超参数搜索对象
            if search_type == 'grid':
                print(f"使用网格搜索进行超参数优化...")
                search = GridSearchCV(
                    model_info['model'],
                    model_info['params'],
                    cv=cv_folds,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
            else:  # random search
                print(f"使用随机搜索进行超参数优化 (迭代次数: {n_iter})...")
                search = RandomizedSearchCV(
                    model_info['model'],
                    model_info['params'],
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring='roc_auc',
                    n_jobs=-1,
                    random_state=42,
                    verbose=1
                )
            
            # 训练模型
            search.fit(X_train, y_train)
            
            # 获取最佳模型
            best_model = search.best_estimator_
            best_models[model_key] = best_model
            
            # 在测试集上评估
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            y_pred = best_model.predict(X_test)
            
            # 计算评估指标
            test_auc = roc_auc_score(y_test, y_pred_proba)
            test_acc = accuracy_score(y_test, y_pred)
            
            # 计算交叉验证分数
            cv_scores = cross_val_score(best_model, X, y, cv=cv_folds, scoring='roc_auc')
            cv_auc = cv_scores.mean()
            cv_auc_std = cv_scores.std()
            
            # 记录结果
            elapsed_time = time.time() - start_time
            
            model_result = {
                'name': model_info['name'],
                'best_params': search.best_params_,
                'test_auc': test_auc,
                'test_acc': test_acc,
                'cv_auc': cv_auc,
                'cv_auc_std': cv_auc_std,
                'training_time': elapsed_time
            }
            
            results[model_key] = model_result
            
            # 添加到指标列表
            all_metrics.append({
                'model': model_info['name'],
                'test_auc': test_auc,
                'test_acc': test_acc,
                'cv_auc': cv_auc,
                'cv_auc_std': cv_auc_std,
                'training_time': elapsed_time
            })
            
            # 打印结果
            print(f"最佳参数: {search.best_params_}")
            print(f"测试集AUC: {test_auc:.4f}, 准确率: {test_acc:.4f}")
            print(f"交叉验证AUC: {cv_auc:.4f} ± {cv_auc_std:.4f}")
            print(f"训练时间: {elapsed_time:.2f}秒")
            
            # 保存模型
            model_path = os.path.join(self.models_dir, f"automl_{model_key}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            print(f"模型已保存到 {model_path}")
        
        # 创建性能比较DataFrame
        performance_df = pd.DataFrame(all_metrics)
        
        # 按AUC降序排序
        performance_df = performance_df.sort_values('cv_auc', ascending=False).reset_index(drop=True)
        
        # 打印性能比较
        print("\n===== 模型性能比较 =====")
        print(performance_df)
        
        # 保存性能比较结果
        performance_path = os.path.join(self.results_dir, 'automl_model_performance.csv')
        performance_df.to_csv(performance_path, index=False)
        print(f"模型性能比较结果已保存到 {performance_path}")
        
        # 绘制模型性能比较图
        self._plot_model_comparison(performance_df)
        
        # 获取最佳模型
        best_model_key = performance_df.iloc[0]['model']
        print(f"\n最佳模型: {best_model_key} (交叉验证AUC: {performance_df.iloc[0]['cv_auc']:.4f})")
        
        return {
            'results': results,
            'best_models': best_models,
            'performance_df': performance_df
        }
    
    def _plot_model_comparison(self, performance_df: pd.DataFrame) -> None:
        """
        绘制模型性能比较图
        
        Args:
            performance_df: 性能比较DataFrame
        """
        # 设置图表样式
        plt.figure(figsize=(12, 8))
        
        # 绘制AUC比较
        plt.subplot(2, 1, 1)
        ax = sns.barplot(x='model', y='cv_auc', data=performance_df)
        
        # 添加误差线
        for i, row in performance_df.iterrows():
            ax.errorbar(i, row['cv_auc'], yerr=row['cv_auc_std'], fmt='none', color='black', capsize=5)
        
        plt.title('模型AUC比较 (交叉验证)', fontsize=14)
        plt.ylabel('AUC分数', fontsize=12)
        plt.ylim(0.5, 1.0)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制训练时间比较
        plt.subplot(2, 1, 2)
        sns.barplot(x='model', y='training_time', data=performance_df)
        plt.title('模型训练时间比较', fontsize=14)
        plt.ylabel('训练时间 (秒)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(
            os.path.join(self.figures_dir, 'automl_model_comparison.png'),
            dpi=300,
            format='png'
        )
        print(f"模型比较图已保存到 {os.path.join(self.figures_dir, 'automl_model_comparison.png')}")
    
    def run(self, cv_folds: int = 5, search_type: str = 'random', n_iter: int = 20) -> Dict[str, Any]:
        """
        运行AutoML流程
        
        Args:
            cv_folds: 交叉验证折数
            search_type: 超参数搜索类型，'grid'或'random'
            n_iter: 随机搜索迭代次数
            
        Returns:
            AutoML结果
        """
        print("开始AutoML流程...")
        
        # 加载数据
        X, y = self.load_data()
        
        # 训练和评估模型
        results = self.train_and_evaluate(X, y, cv_folds, search_type, n_iter)
        
        print("\nAutoML流程完成!")
        return results


# 如果直接运行此脚本，则执行AutoML
if __name__ == "__main__":
    automl = AutoML()
    results = automl.run()