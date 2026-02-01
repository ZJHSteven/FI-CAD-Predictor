import os  # 导入os模块，用于文件和目录操作
import glob  # 导入glob模块，用于文件通配符查找（本脚本未用到）
import joblib  # 导入joblib模块，用于模型的序列化和反序列化
import shap  # 导入shap库，用于模型可解释性分析
import pandas as pd  # 导入pandas库，用于数据处理
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
from utils.config_loader import create_config_loader  # 导入自定义的配置加载器

# 加载配置文件
config_loader = create_config_loader()  # 创建配置加载器实例
paths_config = config_loader.get_paths_config()  # 获取路径相关的配置字典

# 获取模型、结果、图表的目录路径
models_dir = paths_config['output']['pycaret_models_dir']  # 模型保存目录
results_dir = paths_config['output']['results_dir']  # 预测结果保存目录
figures_dir = paths_config['output']['figures_dir']  # 图表保存目录

# 获取所有模型pkl文件（不包含pycaret自动生成的子目录）
model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') ]  # 列出所有以.pkl结尾的模型文件

# 确保图表输出目录存在，不存在则创建
os.makedirs(figures_dir, exist_ok=True)

# 遍历每一个模型文件，依次进行SHAP可解释性分析
for model_file in model_files:
    model_path = os.path.join(models_dir, model_file)  # 拼接出模型文件的完整路径
    model_name = os.path.splitext(model_file)[0]  # 获取模型文件名（去掉扩展名）
    # 修正：去掉_model后缀再拼接_predict.csv
    if model_name.endswith('_model'):
        result_base = model_name[:-6]  # 如果以_model结尾，去掉_model后缀
    else:
        result_base = model_name  # 否则直接用模型名
    result_csv = os.path.join(results_dir, f"{result_base}_predict.csv")  # 拼接预测结果csv文件路径
    if not os.path.exists(result_csv):  # 如果结果文件不存在
        print(f"结果文件不存在: {result_csv}")  # 打印提示
        continue  # 跳过当前模型
    # 读取csv，去掉最后两列（通常为预测标签和概率/分数等）
    df = pd.read_csv(result_csv)  # 读取预测结果csv为DataFrame
    if df.shape[1] <= 2:  # 如果特征列数不足
        print(f"特征列数不足: {result_csv}")  # 打印提示
        continue  # 跳过当前模型
    X = df.iloc[:, :-2]  # 取除最后两列外的所有列作为特征输入
    # 加载模型（含pipeline）
    model = joblib.load(model_path)  # 加载模型文件
    # pipeline特征预处理，只取最后一层estimator
    pipeline = None  # 初始化pipeline变量
    estimator = model  # 初始化estimator为模型本身
    if hasattr(model, 'pipeline'):  # 如果模型有pipeline属性
        pipeline = model.pipeline  # 取出pipeline
    elif hasattr(model, 'steps'):  # 如果模型本身就是pipeline
        pipeline = model
    # 拆分pipeline：只对特征做transform，不包含最后一层estimator
    if pipeline is not None and hasattr(pipeline, 'steps') and len(pipeline.steps) > 1:
        try:
            from sklearn.pipeline import Pipeline  # 导入Pipeline类
            feature_pipeline = Pipeline(pipeline.steps[:-1])  # 只取除最后一步外的所有步骤，作为特征预处理pipeline
            X_trans = feature_pipeline.transform(X)  # 对特征做transform
            estimator = pipeline.steps[-1][1]  # 取出最后一层的estimator
        except Exception as e:
            print(f"特征预处理失败: {model_name}, 错误: {e}")  # 预处理失败则打印错误
            continue  # 跳过当前模型
    else:
        try:
            X_trans = X if not hasattr(estimator, 'transform') else estimator.transform(X)  # 如果estimator有transform方法则直接transform，否则用原始特征
        except Exception as e:
            print(f"特征预处理失败: {model_name}, 错误: {e}")  # 预处理失败则打印错误
            continue  # 跳过当前模型
    # shap explainer
    try:
        explainer = shap.Explainer(estimator, X_trans)  # 创建shap解释器
        shap_values = explainer(X_trans)  # 计算shap值
        # 绘制summary plot
        plt.figure(figsize=(10, 6))  # 新建画布
        shap.summary_plot(shap_values, X_trans, show=False)  # 绘制shap summary图，不直接展示
        fig_path = os.path.join(figures_dir, f"{model_name}_shap_summary.png")  # 拼接图像保存路径
        plt.savefig(fig_path, bbox_inches='tight')  # 保存图片到文件
        plt.close()  # 关闭画布
        print(f"SHAP图已保存: {fig_path}")  # 打印保存成功信息
    except Exception as e:
        print(f"SHAP绘图失败: {model_name}, 错误: {e}")  # 绘图失败则打印错误
