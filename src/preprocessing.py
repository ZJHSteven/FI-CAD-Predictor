# 数据预处理模块 (Data Preprocessing Module)
# 用于数据清洗和预处理
# 本模块实现了一个完整的数据预处理流程，包括数据加载、清洗、转换和特征工程
# 主要功能包括：缺失值处理、异常值替换、特征标准化、哑变量创建、相关性分析等

import os  # 导入操作系统模块，用于文件和目录操作
import pandas as pd  # 导入pandas库，用于数据处理和分析
import numpy as np  # 导入numpy库，用于数值计算
from typing import  List  # 导入类型提示，用于函数参数和返回值的类型注解
from sklearn.preprocessing import StandardScaler  # 导入标准化工具，用于特征标准化
import matplotlib.pyplot as plt  # 导入matplotlib库，用于数据可视化
import seaborn as sns  # 导入seaborn库，用于高级数据可视化

from utils.config_loader import create_config_loader  # 导入配置加载器，用于加载项目配置


class DataPreprocessor:
    """
    数据预处理类，用于数据清洗和预处理
    
    该类实现了一个完整的数据预处理流程，包括：
    1. 数据加载和合并
    2. 数据类型转换
    3. 缺失值处理
    4. 分类变量编码（哑变量创建）
    5. 数值特征标准化
    6. 特征筛选（常数列、高相关特征、低方差特征）
    7. 数据可视化（相关性热图）
    
    所有配置参数都从配置文件中加载，便于调整和实验
    """
    
    def __init__(self, config_loader=None):
        """
        初始化数据预处理器

        只加载路径配置和变量配置，所有路径和参数均来自配置文件。
        支持外部传入config_loader，便于测试和灵活配置。
        """
        # 如果未传入config_loader，则使用默认的create_config_loader
        if config_loader is None:
            self.config_loader = create_config_loader()
        else:
            self.config_loader = config_loader

        # 加载路径相关配置（如原始数据目录、清洗后数据目录、文件名等）
        self.paths_config = self.config_loader.get_paths_config()  # 路径配置字典
        # 加载变量相关配置（如排除列、预处理参数等）
        self.variables_config = self.config_loader.get_variables_config()  # 变量配置字典
        self.viz_config = self.config_loader.get_viz_config()  # 可视化配置字典

        # 路径相关变量
        self.raw_dir = self.paths_config['data']['raw_dir']  # 原始数据目录
        self.cleaned_dir = self.paths_config['data']['cleaned_dir']  # 清洗后数据目录
        self.basic_cleaned_file = self.paths_config['data'].get('basic_cleaned_file')  # 基础清洗后文件名（可选）
        self.standardized_file = self.paths_config['data'].get('standardized_file')  # 标准化后文件名（可选）
        self.onehot_file = self.paths_config['data'].get('onehot_file')  # 哑变量文件名（可选）
        self.standardized_onehot_dropref_file = self.paths_config['data'].get('standardized_onehot_dropref_file')  # 标准化+哑变量文件名（可选）

        # 变量和预处理参数
        self.exclude_columns = self.variables_config['exclude_columns']  # 需要排除的列（如ID、目标变量等）
        self.preprocessing_config = self.variables_config['preprocessing']  # 预处理相关参数（如是否标准化、相关性阈值等）

        # 读取分类变量及映射配置
        self.categorical_mappings = self.variables_config.get('categorical_mappings', {})

        # 创建清洗后数据输出目录（如果不存在则自动创建）
        os.makedirs(self.cleaned_dir, exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """
        加载原始数据
        
        从配置指定的路径加载协变量和结局变量数据文件，并基于ID列合并数据。
        路径直接使用配置文件中的完整相对路径，并转换为绝对路径，保证兼容性。
        
        Returns:
            合并后的数据框
        
        Raises:
            FileNotFoundError: 当协变量或结局变量文件不存在时抛出
        
        工作流程：
        1. 从配置中获取协变量和结局变量文件的完整路径
        2. 转换为绝对路径
        3. 检查文件是否存在
        4. 读取协变量和结局变量数据
        5. 基于ID列合并数据
        6. 删除排除列
        7. 返回合并后的数据框
        """
        # 获取文件路径（直接用配置文件里的完整路径，并转为绝对路径）
        covariates_path = os.path.abspath(self.paths_config['data']['covariates_file'])  # 协变量文件绝对路径
        targets_path = os.path.abspath(self.paths_config['data']['targets_file'])  # 结局变量文件绝对路径

        # 检查文件是否存在
        if not os.path.exists(covariates_path):
            raise FileNotFoundError(f"协变量文件 {covariates_path} 不存在")
        if not os.path.exists(targets_path):
            raise FileNotFoundError(f"结局变量文件 {targets_path} 不存在")

        # 读取数据
        covariates = pd.read_csv(covariates_path)  # 读取协变量数据
        targets = pd.read_csv(targets_path)  # 读取结局变量数据

        # 合并数据
        df = pd.merge(covariates, targets, on='ID')  # 基于ID列合并协变量和结局变量数据

        # 删除排除列（如ID、BMI_to_FI等），保证后续处理不包含这些列
        df = df.drop(columns=self.exclude_columns, errors='ignore')

        print(f"加载数据完成，共 {df.shape[0]} 行，{df.shape[1]} 列（已排除指定列）")
        return df  # 返回合并后的数据框
    
    def replace_abnormal_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        只做异常值替换，不做填充。
        将常见异常值（如inf、-inf、'na'、'NA'、空字符串等）统一替换为np.nan。
        这样后续类型识别和填充不会受异常值干扰。
        """
        df = df.replace([np.inf, -np.inf, 'na', 'NA', '', ' '], np.nan)
        return df

    def convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        自动判断每一列的数据类型并赋值：
        - 唯一值个数≤5的列视为分类型变量（包括二分类、三分类、五分类等），其余为数值型变量。
        - 二分类（唯一值为2）可转为category，3~5分类转为有序category（ordered=True）。
        - 目前所有3~5分类变量均为有序分类，未来如有无序分类变量需单独处理。
        - 自动生成categorical_vars和numeric_vars两个列表，作为类属性保存。
        
        Args:
            df: 输入数据框
        Returns:
            处理后的数据框
        """
        self.categorical_vars = []  # 存储分类型变量名
        self.numeric_vars = []      # 存储数值型变量名
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            n_unique = len(unique_vals)
            # 跳过全空列
            if n_unique == 0:
                continue
            # 二分类
            if n_unique == 2:
                df[col] = df[col].astype('category')
                self.categorical_vars.append(col)
            # 3~5分类（目前均为有序分类，未来如有无序分类变量需单独处理）
            elif 2 < n_unique <= 5:
                df[col] = pd.Categorical(df[col], categories=sorted(unique_vals), ordered=True)
                self.categorical_vars.append(col)
            # 数值型
            else:
                try:
                    df[col] = df[col].astype(float)
                    self.numeric_vars.append(col)
                except Exception:
                    # 转换失败则仍视为分类型
                    df[col] = df[col].astype('category')
                    self.categorical_vars.append(col)
        print(f"自动识别分类型变量: {self.categorical_vars}")
        print(f"自动识别数值型变量: {self.numeric_vars}")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        缺失值填充，依赖于categorical_vars和numeric_vars，必须在类型识别后调用。
        数值型变量用中位数填充，分类型变量用众数填充。
        """
        # 数值型变量用中位数填充
        if hasattr(self, 'numeric_vars') and self.numeric_vars:
            df[self.numeric_vars] = df[self.numeric_vars].fillna(df[self.numeric_vars].median())
        # 分类型变量用众数填充
        if hasattr(self, 'categorical_vars') and self.categorical_vars:
            for col in self.categorical_vars:
                if df[col].isnull().any():
                    mode = df[col].mode()
                    if not mode.empty:
                        df[col] = df[col].fillna(mode.iloc[0])
        return df

    def create_dummy_variables(self, df: pd.DataFrame, drop_reference: bool = True) -> pd.DataFrame:
        """
        创建哑变量（独热编码），所有映射和参考组均来自配置文件。
        Args:
            df: 输入数据框
            drop_reference: 是否删除参考组
        Returns:
            处理后的数据框
        """
        columns_to_drop = []
        for col, info in self.categorical_mappings.items():
            mapping = info.get('mapping', {})
            reference = info.get('reference', None)
            if col in df.columns:
                df[col] = df[col].map(mapping)
                dummies = pd.get_dummies(df[col], prefix=col)
                # 删除参考组
                if drop_reference and reference is not None:
                    ref_val = mapping.get(reference, reference)
                    ref_col = f'{col}_{ref_val}'
                    if ref_col in dummies.columns:
                        dummies = dummies.drop(ref_col, axis=1)
                df = pd.concat([df, dummies], axis=1)
                columns_to_drop.append(col)
        df = df.drop(columns_to_drop, axis=1, errors='ignore')
        return df

    def standardize_features(self, df: pd.DataFrame, numeric_vars: List[str]) -> pd.DataFrame:
        """
        标准化数值特征
        
        使用StandardScaler将数值特征标准化为均值为0、标准差为1的分布
        
        Args:
            df: 输入数据框
            numeric_vars: 数值型变量列表
            
        Returns:
            处理后的数据框
            
        工作流程：
        1. 检查是否需要标准化（根据配置）
        2. 如果需要，使用StandardScaler对数值特征进行标准化
        3. 返回处理后的数据框
        """
        if self.preprocessing_config['standardize']:  # 如果配置中启用了标准化
            scaler = StandardScaler()  # 创建标准化器
            df[numeric_vars] = scaler.fit_transform(df[numeric_vars])  # 对数值特征进行标准化
        
        return df  # 返回处理后的数据框
    
    def remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        删除常数列
        
        删除数据框中只有一个唯一值的列（常数列）
        
        Args:
            df: 输入数据框
            
        Returns:
            处理后的数据框
            
        工作流程：
        1. 检查是否需要删除常数列（根据配置）
        2. 如果需要，找出唯一值数量小于等于1的列
        3. 删除这些列
        4. 返回处理后的数据框
        """
        if self.preprocessing_config['remove_constant']:  # 如果配置中启用了删除常数列
            const_cols = [col for col in df.columns if df[col].nunique() <= 1]  # 找出唯一值数量小于等于1的列
            if const_cols:  # 如果存在常数列
                print(f"发现{len(const_cols)}个常数列，将被移除: {const_cols}")  # 打印常数列信息
                df = df.drop(columns=const_cols)  # 删除常数列
        
        return df  # 返回处理后的数据框
    
    def remove_correlated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        删除高度相关的特征
        
        计算特征之间的相关系数，删除与其他特征高度相关的特征
        
        Args:
            df: 输入数据框
            
        Returns:
            处理后的数据框
            
        工作流程：
        1. 检查是否需要删除高度相关特征（根据配置）
        2. 如果需要，获取特征列（排除目标变量和ID）
        3. 计算特征之间的相关系数矩阵
        4. 获取相关系数矩阵的上三角矩阵（避免重复计算）
        5. 找出相关系数大于阈值的特征
        6. 删除这些特征
        7. 返回处理后的数据框
        """
        if self.preprocessing_config['remove_correlated']:  # 如果配置中启用了删除高度相关特征
            # 获取特征列（排除目标变量和ID）
            feature_cols = [col for col in df.columns if col not in self.exclude_columns]  # 获取特征列
            
            # 计算相关系数矩阵
            corr_matrix = df[feature_cols].corr().abs()  # 计算特征之间的绝对相关系数
            
            # 获取相关系数矩阵的上三角矩阵
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # 获取上三角矩阵
            
            # 找出相关系数大于阈值的特征
            threshold = self.preprocessing_config['correlation_threshold']  # 获取相关系数阈值
            highly_corr_features = [column for column in upper.columns if any(upper[column] > threshold)]  # 找出高度相关特征
            
            if highly_corr_features:  # 如果存在高度相关特征
                print(f"发现{len(highly_corr_features)}个高度相关的特征，将被移除: {highly_corr_features}")  # 打印高度相关特征信息
                df = df.drop(columns=highly_corr_features)  # 删除高度相关特征
        
        return df  # 返回处理后的数据框
    
    def remove_low_variance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        删除低方差特征
        
        计算特征的方差，删除方差低于阈值的特征
        
        Args:
            df: 输入数据框
            
        Returns:
            处理后的数据框
            
        工作流程：
        1. 检查是否需要删除低方差特征（根据配置）
        2. 如果需要，导入VarianceThreshold
        3. 获取特征列（排除目标变量和ID）
        4. 计算特征的方差
        5. 找出方差低于阈值的特征
        6. 删除这些特征
        7. 返回处理后的数据框
        """
        if self.preprocessing_config['remove_low_variance']:  # 如果配置中启用了删除低方差特征
            from sklearn.feature_selection import VarianceThreshold  # 导入方差阈值选择器
            
            # 获取特征列（排除目标变量和ID）
            feature_cols = [col for col in df.columns if col not in self.exclude_columns]  # 获取特征列
            
            # 计算方差
            var_threshold = VarianceThreshold(threshold=self.preprocessing_config['variance_threshold'])  # 创建方差阈值选择器
            var_threshold.fit(df[feature_cols])  # 拟合方差阈值选择器
            
            # 找出低方差特征
            low_var_features = [column for column, variance in zip(feature_cols, var_threshold.variances_) 
                               if variance <= self.preprocessing_config['variance_threshold']]  # 找出低方差特征
            
            if low_var_features:  # 如果存在低方差特征
                print(f"发现{len(low_var_features)}个低方差特征，将被移除: {low_var_features}")  # 打印低方差特征信息
                df = df.drop(columns=low_var_features)  # 删除低方差特征
        
        return df  # 返回处理后的数据框
    
    def plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """
        绘制相关性热图
        
        计算特征之间的相关系数，并绘制热图可视化
        
        Args:
            df: 输入数据框
            
        工作流程：
        1. 获取特征列（排除目标变量和ID）
        2. 设置中文显示
        3. 绘制热图
        4. 设置标题和标签
        5. 保存图片
        """
        # 获取特征列（排除目标变量和ID）
        feature_cols = [col for col in df.columns if col not in self.exclude_columns]  # 获取特征列
        
        # 设置中文显示
        plt.rcParams['font.sans-serif'] = [self.viz_config['global']['font_family']]  # 设置字体
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        
        # 绘制热图
        plt.figure(figsize=tuple(self.viz_config['global']['figure_size']))  # 设置图形大小
        sns.heatmap(
            df[feature_cols].corr(),  # 计算相关系数矩阵
            cmap=self.viz_config['correlation']['color_map'],  # 设置颜色映射
            center=self.viz_config['correlation']['center'],  # 设置中心值
            annot=self.viz_config['correlation']['annot']  # 是否显示数值
        )
        plt.title(self.viz_config['correlation']['title'])  # 设置标题
        plt.xticks(rotation=self.viz_config['correlation']['xticklabels_rotation'], ha='right')  # 设置x轴标签旋转
        plt.yticks(rotation=self.viz_config['correlation']['yticklabels_rotation'])  # 设置y轴标签旋转
        plt.tight_layout()  # 调整布局
        
        # 保存图片
        os.makedirs(self.paths_config['output']['figures_dir'], exist_ok=True)  # 创建图片目录
        plt.savefig(
            os.path.join(self.paths_config['output']['figures_dir'], 'correlation_heatmap.png'),  # 图片路径
            dpi=self.viz_config['global']['dpi'],  # 设置DPI
            format=self.viz_config['global']['save_format']  # 设置保存格式
        )
    
    def preprocess(self) -> None:
        """
        执行完整的预处理流程
        1. 加载数据
        2. 异常值统一替换为NaN
        3. 自动识别变量类型并赋值
        4. 缺失值填充
        5. 创建哑变量
        6. 标准化特征
        7. 删除常数列
        8. 删除高度相关的特征
        9. 删除低方差特征
        10. 绘制相关性热图
        11. 保存清洗后的数据
        12. 返回处理后的数据框
        """
        print("开始数据预处理...")
        # 1. 基础清洗流程
        df = self.load_data()  # 加载原始数据并合并
        df = self.replace_abnormal_values(df)  # 将异常值统一替换为NaN
        df = self.convert_to_numeric(df)  # 自动识别变量类型（数值型/分类型）
        df = self.handle_missing_values(df)  # 缺失值填充（数值型用中位数，分类型用众数）
        df = self.remove_constant_columns(df)  # 删除常数列（只有一个唯一值的列）
        df = self.remove_correlated_features(df)  # 删除高度相关的特征
        df = self.remove_low_variance_features(df)  # 删除低方差特征
        self.plot_correlation_heatmap(df)  # 绘制并保存相关性热图
        # 保存基础清洗文件
        basic_cleaned_path = os.path.abspath(self.basic_cleaned_file)
        df.to_csv(basic_cleaned_path, index=False)
        print(f"基础清洗数据已保存到 {basic_cleaned_path}")

        # 2. 标准化文件（只做标准化，不做哑变量）
        df_std = self.standardize_features(df.copy(), self.numeric_vars)
        standardized_path = os.path.abspath(self.standardized_file)
        df_std.to_csv(standardized_path, index=False)
        print(f"标准化数据已保存到 {standardized_path}")

        # 3. 哑变量文件（不删除参考列，适合树模型）
        df_onehot = self.create_dummy_variables(df.copy(), drop_reference=False)
        onehot_path = os.path.abspath(self.onehot_file)
        df_onehot.to_csv(onehot_path, index=False)
        print(f"哑变量数据（保留参考列）已保存到 {onehot_path}")

        # 4. 标准化+哑变量+删除参考列文件（适合逻辑回归/MLP等）
        df_onehot_dropref = self.create_dummy_variables(df.copy(), drop_reference=True)
        df_onehot_dropref = self.standardize_features(df_onehot_dropref, [col for col in df_onehot_dropref.columns if col in self.numeric_vars])
        std_onehot_dropref_path = os.path.abspath(self.standardized_onehot_dropref_file)
        df_onehot_dropref.to_csv(std_onehot_dropref_path, index=False)
        print(f"标准化+哑变量（删除参考列）数据已保存到 {std_onehot_dropref_path}")

        print("全部数据预处理文件已生成。")
        return None


# 如果直接运行此脚本，则执行预处理
if __name__ == "__main__":
    preprocessor = DataPreprocessor()  # 创建数据预处理器实例
    preprocessor.preprocess()  # 执行预处理