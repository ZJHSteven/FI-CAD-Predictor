import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any

import yaml  # 新增：导入yaml库


# 导入PyCaret库
from pycaret.classification import *

# 导入配置加载器
from utils.config_loader import create_config_loader

# 禁用PyCaret的详细日志输出
import os
os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"  




class PyCaretAutoML:
    """
    使用PyCaret库实现AutoML流程，用于FI预测CVD的数学建模项目
    """
    
    def __init__(self):
        """
        初始化AutoML类，只保留核心路径配置，全部从paths.yaml读取。
        """
        # 1. 加载配置文件（paths.yaml），获取所有路径配置
        self.config_loader = create_config_loader()
        self.config = self.config_loader.get_paths_config()

        # 2. 获取项目根目录（以project为根，便于拼接绝对路径）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 新增：加载pycaret_config.yaml配置
        self.pycaret_config = self.config_loader.load_config("pycaret_config")
        self.pycaret_model_config = os.path.join(project_root, self.config['configures']['pycaret_model_config'])

        # 3. 数据文件路径：只用基础清洗后的数据（basic_cleaned_file）
        self.data_file = os.path.join(project_root, self.config['data']['basic_cleaned_file'])
        # 4. 模型保存目录（pycaret模型子目录）
        self.pycaret_models_dir = os.path.join(project_root, self.config['output']['pycaret_models_dir'])
        # 5. 图表输出目录（pycaret图表子目录）
        self.pycaret_figures_dir = os.path.join(project_root, self.config['output']['figures_dir'], 'pycaret')
        # 6. 结果输出目录
        self.results_dir = os.path.join(project_root, self.config['output']['results_dir'])

        # 7. 确保所有输出目录存在
        os.makedirs(self.pycaret_models_dir, exist_ok=True)
        os.makedirs(self.pycaret_figures_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # 8. 打印路径信息，便于调试和后续维护
        print(f"初始化完成，将使用PyCaret库进行AutoML建模")
        print(f"数据文件路径: {self.data_file}")
        print(f"PyCaret模型保存目录: {self.pycaret_models_dir}")
        print(f"PyCaret图表保存目录: {self.pycaret_figures_dir}")
        print(f"结果输出目录: {self.results_dir}")
        print(f"PyCaret配置: {self.pycaret_config}")


    
    def load_cleaned_data(self) -> pd.DataFrame:
        """
        直接读取清洗好的数据文件（basic_cleaned_data.csv），只返回数据。
        Returns:
            data: pd.DataFrame, 预处理好的数据
        """
        data = pd.read_csv(self.data_file, encoding='utf-8')
        print(f"已读取清洗后数据，形状: {data.shape}")
        print(f"数据所有列: {list(data.columns)}")
        if 'FI' in data.columns:
            print(f"FI 列存在，描述统计如下：\n{data['FI'].describe()}")
        else:
            print("FI 列不存在于数据中！")
        return data


    def run_pycaret(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        使用PyCaret进行AutoML建模
        
        Args:
            data: 预处理后的数据
            target_col: 目标变量列名
            
        Returns:
            PyCaret模型结果字典
        """
        print("\n===== 开始PyCaret AutoML建模 =====")
        print(f"建模前数据所有列: {list(data.columns)}")
        if 'FI' in data.columns:
            print(f"建模前 FI 列描述统计：\n{data['FI'].describe()}")
        else:
            print("建模前数据中 FI 列不存在！")
        # 设置实验
        print("初始化PyCaret实验...")
        clf = setup(data=data, **self.pycaret_config['setup'])
        print("PyCaret实验初始化完成")
        print(f"PyCaret setup 后特征列: {list(get_config('X').columns)}")
        if 'FI' in get_config('X').columns:
            print("PyCaret setup 后，FI 作为特征被保留！")
        else:
            print("PyCaret setup 后，FI 没有被用作特征！")

        # 模型比较
        print("\n比较所有模型...")
        best_models = compare_models(**self.pycaret_config['compare'])
        print(f"模型比较完成，已选出前{self.pycaret_config['compare']['n_select']}个模型")
        # 兼容单模型和多模型返回
        if not isinstance(best_models, list):
            best_models = [best_models]
        all_models = best_models
        
        results = {}
        tuned_models = {}
        all_model_params = {}
        
        # 对每个模型进行调优和保存
        print("\n对选定的模型进行调优和保存...")
        for model in all_models:
            model_name = model.__class__.__name__
            print(f"\n===== 开始调优模型: {model_name} =====")
            try:
                tuned_model = tune_model(model, **self.pycaret_config['tune'])
                print(f"调优完成: {model_name}")
            except Exception as e:
                print(f"调优模型 {model_name} 时出错: {e}")
                continue
            tuned_models[model_name] = tuned_model
            
            # 保存模型
            model_filename = f"pycaret_{model_name.lower().replace(' ', '_')}_model"
            model_path = os.path.join(self.pycaret_models_dir, f"{model_filename}")
            save_model(tuned_model, model_path)
            print(f"模型已保存到: {model_path}")
            
            # 收集模型超参数
            param_dict = tuned_model.get_params()
            all_model_params[model_name.lower()] = param_dict
            
            # 所有模型参数一次性写入 pycaret_model_config.yaml（只写模型参数，不嵌套 models 节点）
            with open(self.pycaret_model_config, 'w', encoding='utf-8') as f:
                yaml.dump(all_model_params, f, allow_unicode=True)
            print(f"所有模型超参数已集中保存到 pycaret_model_config.yaml")

            # 记录结果
            results[model_name] = {
                'model': tuned_model,
                'params': param_dict,
                'auc': pull().loc[pull().index[0], 'AUC'],
                'accuracy': pull().loc[pull().index[0], 'Accuracy'],
                'recall': pull().loc[pull().index[0], 'Recall'],
                'precision': pull().loc[pull().index[0], 'Prec.'],
                'f1': pull().loc[pull().index[0], 'F1'],
                'kappa': pull().loc[pull().index[0], 'Kappa']
            }

            # 获取PyCaret自动处理后的特征
            X_final = get_config('X')  # 这是PyCaret setup后自动筛选、预处理的特征DataFrame

            # 用当前调优好的模型做预测
            predict_df = predict_model(tuned_model, data=X_final)

            # 保存预测结果
            predict_csv_path = os.path.join(self.results_dir, f"pycaret_{model_name.lower()}_predict.csv")
            predict_df.to_csv(predict_csv_path, index=False, encoding='utf-8')
            print(f"{model_name}预测结果已保存到: {predict_csv_path}")
        
        # 结果汇总保存为csv
        metrics_csv_path = os.path.join(self.results_dir, 'pycaret_model_metrics.csv')
        metrics_list = []
        for model_name, res in results.items():
            metrics_list.append({
                'Model': model_name,
                'AUC': res['auc'],
                'Accuracy': res['accuracy'],
                'Recall': res['recall'],
                'Precision': res['precision'],
                'F1': res['f1'],
                'Kappa': res['kappa']
            })
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(metrics_csv_path, index=False, encoding='utf-8')
        print(f"所有模型主要指标已保存为csv: {metrics_csv_path}")
        
        # 生成可视化结果
        self._create_visualizations(tuned_models, results)
        
        return {
            'setup': clf,
            'models': tuned_models,
            'results': results
        }
    
    def _create_visualizations(self, models: Dict, results: Dict) -> None:
        """
        创建模型可视化结果
        
        Args:
            models: 调优后的模型字典
            results: 模型结果字典
        """
        print("\n===== 创建模型可视化结果 =====")
        # 设置matplotlib中文字体为黑体，防止中文乱码
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 1. 模型性能比较图
        print("生成模型性能比较图...")
        plt.figure(figsize=(12, 8))
        
        # 准备数据
        model_names = list(results.keys())
        metrics = ['auc', 'accuracy', 'f1', 'precision', 'recall']
        
        # 创建性能比较DataFrame
        performance_data = []
        for model_name in model_names:
            for metric in metrics:
                performance_data.append({
                    'Model': model_name,
                    'Metric': metric.upper(),
                    'Value': results[model_name][metric]
                })
        
        performance_df = pd.DataFrame(performance_data)
        
        # 绘制性能比较图
        sns.barplot(x='Model', y='Value', hue='Metric', data=performance_df)
        plt.title('模型性能比较', fontsize=15)
        plt.xlabel('模型', fontsize=12)
        plt.ylabel('性能指标值', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='指标')
        plt.tight_layout()
        
        # 保存图表
        performance_path = os.path.join(self.pycaret_figures_dir, 'pycaret_model_performance.png')
        plt.savefig(performance_path, dpi=300)
        print(f"模型性能比较图已保存到: {performance_path}")
        
        import shutil

        # 2. ROC曲线
        print("生成ROC曲线...")
        for model_name, model in models.items():
            try:
                plot_model(model, plot='auc', save=True)
                src = 'AUC.png'
                dst = os.path.join(self.pycaret_figures_dir, f'pycaret_{model_name.lower().replace(" ", "_")}_roc.png')
                shutil.move(src, dst)
                print(f"ROC曲线已保存: {dst}")
            except Exception as e:
                print(f"无法为 {model_name} 生成ROC图: {str(e)}")

        # 3. 混淆矩阵
        print("生成混淆矩阵...")
        for model_name, model in models.items():
            try:
                plot_model(model, plot='confusion_matrix', save=True)
                src = 'Confusion Matrix.png'
                dst = os.path.join(self.pycaret_figures_dir, f'pycaret_{model_name.lower().replace(" ", "_")}_cm.png')
                shutil.move(src, dst)
                print(f"混淆矩阵已保存: {dst}")
            except Exception as e:
                print(f"无法为 {model_name} 生成混淆矩阵图: {str(e)}")

        # 4. 特征重要性
        print("生成特征重要性图...")
        for model_name, model in models.items():
            try:
                plot_model(model, plot='feature', save=True)
                src = 'Feature Importance.png'
                dst = os.path.join(self.pycaret_figures_dir, f'pycaret_{model_name.lower().replace(" ", "_")}_feature.png')
                shutil.move(src, dst)
                print(f"特征重要性图已保存: {dst}")
            except Exception as e:
                print(f"无法为 {model_name} 生成特征重要性图: {str(e)}")

def main():
    """
    主函数
    """
    print("===== 开始PyCaret AutoML流程 =====")
    # 初始化AutoML类
    automl = PyCaretAutoML()
    # 直接读取清洗好的数据
    data= automl.load_cleaned_data()
        # 运行PyCaret AutoML
    results = automl.run_pycaret(data)
    print("\n===== PyCaret AutoML流程完成 =====")
    print(f"所有模型和结果已保存到指定目录")

if __name__ == "__main__":
    from utils.logger import StdoutRedirector
    StdoutRedirector()
    main()