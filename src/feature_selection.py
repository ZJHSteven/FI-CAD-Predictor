# 特征选择模块
# 用于筛选最优变量集

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFECV, SelectKBest, f_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.utils.config_loader import create_config_loader


class FeatureSelector:
    """
    特征选择类，用于筛选最优变量集
    """
    
    def __init__(self, config_loader=None):
        """
        初始化特征选择器
        
        Args:
            config_loader: 配置加载器实例，如果为None则创建新实例
        """
        if config_loader is None:
            self.config_loader = create_config_loader()
        else:
            self.config_loader = config_loader
        
        # 加载配置
        self.paths_config = self.config_loader.get_paths_config()
        self.variables_config = self.config_loader.get_variables_config()
        self.viz_config = self.config_loader.get_viz_config()
        
        # 设置路径
        self.cleaned_dir = self.paths_config['data']['cleaned_dir']
        self.selected_features_dir = self.paths_config['data']['selected_features_dir']
        
        # 设置排除列
        self.exclude_columns = self.variables_config['exclude_columns']
        
        # 特征选择设置
        self.feature_selection_config = self.variables_config['feature_selection']
        
        # 创建输出目录
        os.makedirs(self.selected_features_dir, exist_ok=True)
        os.makedirs(self.paths_config['output']['figures_dir'], exist_ok=True)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        加载清洗后的数据
        
        Returns:
            特征矩阵和目标变量
        """
        # 获取文件路径
        cleaned_path = os.path.join(self.cleaned_dir, self.paths_config['data']['cleaned_file'])
        
        # 检查文件是否存在
        if not os.path.exists(cleaned_path):
            raise FileNotFoundError(f"清洗后的数据文件 {cleaned_path} 不存在")
        
        # 读取数据
        df = pd.read_csv(cleaned_path)
        
        # 创建目标变量
        y = df['CVD']
        
        # 获取特征矩阵（排除目标变量和其他不需要的列）
        X = df.drop(self.exclude_columns, axis=1, errors='ignore')
        
        print(f"加载数据完成，特征矩阵形状: {X.shape}，目标变量形状: {y.shape}")
        return X, y
    
    def lasso_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        使用LassoCV进行特征选择
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            选择的特征列表
        """
        if not self.feature_selection_config['use_lasso']:
            return []
        
        print("\n===== 方法1: LassoCV 特征选择 =====")
        try:
            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 配置LassoCV
            lasso_config = self.feature_selection_config['lasso']
            lasso = LassoCV(
                cv=lasso_config['cv_folds'],
                random_state=lasso_config['random_state'],
                max_iter=lasso_config['max_iter']
            )
            
            # 拟合模型
            lasso.fit(X_scaled, y)
            
            # 获取特征重要性
            lasso_importance = pd.DataFrame({
                '特征': X.columns,
                '系数': np.abs(lasso.coef_)
            })
            lasso_importance = lasso_importance.sort_values('系数', ascending=False)
            
            # 选择非零系数的特征
            lasso_selected = lasso_importance[lasso_importance['系数'] > 0]['特征'].tolist()
            
            print(f"LassoCV 选择的特征数量: {len(lasso_selected)}")
            print("前10个重要特征:")
            print(lasso_importance.head(10))
            
            # 绘制特征重要性图
            self._plot_feature_importance(
                lasso_importance, 
                'lasso_feature_importance.png', 
                'LassoCV 特征重要性'
            )
            
            return lasso_selected
        
        except Exception as e:
            print(f"LassoCV 特征选择出错: {e}")
            return []
    
    def rfe_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        使用递归特征消除进行特征选择
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            选择的特征列表
        """
        if not self.feature_selection_config['use_rfe']:
            return []
        
        print("\n===== 方法2: 递归特征消除 (RFE) =====")
        try:
            # 配置RFE
            rfe_config = self.feature_selection_config['rfe']
            
            # 创建估计器
            if rfe_config['estimator'] == 'RandomForestClassifier':
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                raise ValueError(f"不支持的估计器: {rfe_config['estimator']}")
            
            # 创建RFECV
            rfecv = RFECV(
                estimator=estimator,
                step=rfe_config['step'],
                cv=StratifiedKFold(rfe_config['cv_folds']),
                scoring=rfe_config['scoring']
            )
            
            # 拟合模型
            rfecv.fit(X, y)
            
            # 获取选择的特征
            rfe_selected = X.columns[rfecv.support_].tolist()
            
            # 获取特征重要性排名
            rfe_importance = pd.DataFrame({
                '特征': X.columns,
                '排名': rfecv.ranking_
            })
            rfe_importance = rfe_importance.sort_values('排名')
            
            print(f"RFE 选择的特征数量: {len(rfe_selected)}")
            print("前10个重要特征:")
            print(rfe_importance.head(10))
            
            # 绘制特征重要性图
            self._plot_feature_importance(
                rfe_importance.rename(columns={'排名': '重要性'}),
                'rfe_feature_importance.png',
                'RFE 特征重要性（排名越低越重要）',
                ascending=True
            )
            
            return rfe_selected
        
        except Exception as e:
            print(f"RFE 特征选择出错: {e}")
            return []
    
    def random_forest_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        使用随机森林特征重要性进行特征选择
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            选择的特征列表
        """
        if not self.feature_selection_config['use_random_forest']:
            return []
        
        print("\n===== 方法3: 随机森林特征重要性 =====")
        try:
            # 配置随机森林
            rf_config = self.feature_selection_config['random_forest']
            rf = RandomForestClassifier(
                n_estimators=rf_config['n_estimators'],
                random_state=rf_config['random_state']
            )
            
            # 拟合模型
            rf.fit(X, y)
            
            # 获取特征重要性
            rf_importance = pd.DataFrame({
                '特征': X.columns,
                '重要性': rf.feature_importances_
            })
            rf_importance = rf_importance.sort_values('重要性', ascending=False)
            
            # 选择重要性大于阈值的特征
            if rf_config['importance_threshold'] == 'mean':
                threshold = rf_importance['重要性'].mean()
            else:
                threshold = float(rf_config['importance_threshold'])
            
            rf_selected = rf_importance[rf_importance['重要性'] > threshold]['特征'].tolist()
            
            print(f"随机森林选择的特征数量: {len(rf_selected)}")
            print("前10个重要特征:")
            print(rf_importance.head(10))
            
            # 绘制特征重要性图
            self._plot_feature_importance(
                rf_importance,
                'rf_feature_importance.png',
                '随机森林特征重要性'
            )
            
            return rf_selected
        
        except Exception as e:
            print(f"随机森林特征选择出错: {e}")
            return []
    
    def p_value_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        使用基于p值的特征选择
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            选择的特征列表
        """
        if not self.feature_selection_config['use_p_value']:
            return []
        
        print("\n===== 方法4: 基于p值的特征选择 =====")
        try:
            # 确保所有数据都是数值类型
            X = X.astype(float)
            
            # 配置p值选择
            p_value_config = self.feature_selection_config['p_value']
            
            # 使用SelectKBest进行单变量特征选择
            print("使用SelectKBest进行单变量特征选择...")
            selector = SelectKBest(f_classif, k='all')
            selector.fit(X, y)
            
            # 获取p值
            p_values = pd.Series(selector.pvalues_, index=X.columns)
            p_value_df = pd.DataFrame({
                '特征': p_values.index,
                'p值': p_values.values
            })
            p_value_df = p_value_df.sort_values('p值')
            
            # 选择p值小于阈值的特征
            threshold = p_value_config['threshold']
            p_selected = p_value_df[p_value_df['p值'] < threshold]['特征'].tolist()
            
            print(f"p值筛选选择的特征数量: {len(p_selected)}")
            print("前10个显著特征:")
            print(p_value_df.head(10))
            
            # 绘制p值图
            self._plot_feature_importance(
                p_value_df.rename(columns={'p值': '重要性'}),
                'p_value_feature_importance.png',
                'p值特征重要性（p值越小越重要）',
                ascending=True
            )
            
            return p_selected
        
        except Exception as e:
            print(f"基于p值的特征选择出错: {e}")
            return []
    
    def _plot_feature_importance(self, importance_df: pd.DataFrame, filename: str, title: str, ascending: bool = False) -> None:
        """
        绘制特征重要性图
        
        Args:
            importance_df: 特征重要性数据框
            filename: 保存的文件名
            title: 图表标题
            ascending: 是否升序排序
        """
        # 设置中文显示
        plt.rcParams['font.sans-serif'] = [self.viz_config['global']['font_family']]
        plt.rcParams['axes.unicode_minus'] = False
        
        # 获取配置
        feature_importance_config = self.viz_config['feature_importance']
        
        # 获取前N个特征
        top_n = min(feature_importance_config['top_n'], len(importance_df))
        if 'p值' in importance_df.columns:
            importance_col = 'p值'
        elif '排名' in importance_df.columns:
            importance_col = '排名'
        else:
            importance_col = '重要性'
        
        # 排序并获取前N个
        df_plot = importance_df.sort_values(importance_col, ascending=ascending).head(top_n)
        
        # 绘制条形图
        plt.figure(figsize=tuple(self.viz_config['global']['figure_size']))
        
        if feature_importance_config['horizontal']:
            bars = plt.barh(df_plot['特征'], df_plot[importance_col])
            plt.xlabel(feature_importance_config['xlabel'])
            plt.ylabel(feature_importance_config['ylabel'])
        else:
            bars = plt.bar(df_plot['特征'], df_plot[importance_col])
            plt.xlabel(feature_importance_config['ylabel'])
            plt.ylabel(feature_importance_config['xlabel'])
            plt.xticks(rotation=45, ha='right')
        
        # 设置标题
        plt.title(title)
        
        # 显示网格
        if self.viz_config['global']['show_grid']:
            plt.grid(True, axis='x' if feature_importance_config['horizontal'] else 'y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(
            os.path.join(self.paths_config['output']['figures_dir'], filename),
            dpi=self.viz_config['global']['dpi'],
            format=self.viz_config['global']['save_format']
        )
    
    def select_features(self) -> pd.DataFrame:
        """
        执行完整的特征选择流程
        
        Returns:
            选择的特征数据框
        """
        print("开始特征选择...")
        
        # 加载数据
        X, y = self.load_data()
        
        # 使用不同方法进行特征选择
        lasso_selected = self.lasso_feature_selection(X, y)
        rfe_selected = self.rfe_feature_selection(X, y)
        rf_selected = self.random_forest_feature_selection(X, y)
        p_selected = self.p_value_feature_selection(X, y)
        
        # 统计每个特征被选择的次数
        all_selected_features = [lasso_selected, rfe_selected, rf_selected, p_selected]
        all_selected_names = ['LassoCV', 'RFE', '随机森林', 'p值筛选']
        
        # 统计每个特征被选择的次数
        feature_counts = Counter()
        for features in all_selected_features:
            feature_counts.update(features)
        
        # 创建特征选择频率DataFrame
        feature_selection_df = pd.DataFrame({
            '特征': list(feature_counts.keys()),
            '选择频率': list(feature_counts.values())
        })
        feature_selection_df = feature_selection_df.sort_values('选择频率', ascending=False)
        
        # 创建每种方法的选择结果
        for method, features in zip(all_selected_names, all_selected_features):
            feature_selection_df[method] = feature_selection_df['特征'].apply(lambda x: 1 if x in features else 0)
        
        # 打印结果
        print("\n===== 特征选择结果 =====")
        print(f"共有 {len(feature_selection_df)} 个特征被至少一种方法选择")
        print("\n被选择频率最高的特征:")
        print(feature_selection_df.head(20))
        
        # 保存结果
        features_path = os.path.join(self.selected_features_dir, self.paths_config['data']['features_file'])
        feature_selection_df.to_csv(features_path, index=False)
        print(f"特征选择结果已保存到 {features_path}")
        
        # 绘制特征选择频率图
        plt.figure(figsize=tuple(self.viz_config['global']['figure_size']))
        top_n = min(20, len(feature_selection_df))
        plt.barh(feature_selection_df['特征'].head(top_n), feature_selection_df['选择频率'].head(top_n))
        plt.xlabel('被选择的方法数')
        plt.ylabel('特征')
        plt.title('特征选择频率 Top20')
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.paths_config['output']['figures_dir'], 'feature_selection_frequency.png'),
            dpi=self.viz_config['global']['dpi'],
            format=self.viz_config['global']['save_format']
        )
        
        # 创建最终特征集
        # 选择被至少两种方法选择的特征
        final_features = feature_selection_df[feature_selection_df['选择频率'] >= 2]['特征'].tolist()
        
        # 如果最终特征集为空，则使用被至少一种方法选择的特征
        if not final_features:
            final_features = feature_selection_df[feature_selection_df['选择频率'] >= 1]['特征'].tolist()
        
        print(f"\n最终选择的特征数量: {len(final_features)}")
        print("最终特征集:")
        print(final_features)
        
        # 创建包含最终特征的数据框
        final_df = pd.concat([X[final_features], y], axis=1)
        
        # 保存最终特征数据
        final_features_path = os.path.join(self.selected_features_dir, 'final_features.csv')
        final_df.to_csv(final_features_path, index=False)
        print(f"最终特征数据已保存到 {final_features_path}")
        
        return final_df


# 如果直接运行此脚本，则执行特征选择
if __name__ == "__main__":
    selector = FeatureSelector()
    final_df = selector.select_features()
    print(f"最终特征数据形状: {final_df.shape}")