import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from utils.config_loader import create_config_loader
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

# 只处理 RidgeClassifier 模型
model_filename = 'pycaret_ridgeclassifier_model.pkl'
result_csv = 'pycaret_ridgeclassifier_predict.csv'

# 加载配置
config_loader = create_config_loader()
paths_config = config_loader.get_paths_config()
models_dir = paths_config['output']['pycaret_models_dir']
results_dir = paths_config['output']['results_dir']
figures_dir = paths_config['output']['figures_dir']

# 路径拼接
model_path = os.path.join(models_dir, model_filename)
result_csv_path = os.path.join(results_dir, result_csv)

# 确保输出目录存在
os.makedirs(figures_dir, exist_ok=True)

# 加载模型
model = joblib.load(model_path)
if not hasattr(model, 'feature_names_in_'):
    raise AttributeError('模型中未找到 feature_names_in_ 属性')
feature_names = list(model.feature_names_in_)

# 读取 result 目录下的 csv
if not os.path.exists(result_csv_path):
    raise FileNotFoundError(f"结果数据集不存在: {result_csv_path}")
df = pd.read_csv(result_csv_path)

# 打印模型和数据相关信息
print(f"模型类型: {type(model)}")
print(f"模型特征名 feature_names_in_: {getattr(model, 'feature_names_in_', None)}")
print(f"模型 coef_ shape: {getattr(model, 'coef_', None).shape if hasattr(model, 'coef_') else '无'}")
print(f"结果数据列: {list(df.columns)}")

# 检查CVD列是否存在，不存在则从原始数据补齐
if 'CVD' not in df.columns:
    raw_cvd_path = os.path.join('data', 'raw', '2011年FI+CVD及变量.csv')
    if not os.path.exists(raw_cvd_path):
        raise FileNotFoundError(f"缺少CVD列且未找到原始数据: {raw_cvd_path}")
    raw_df = pd.read_csv(raw_cvd_path)
    if 'CVD' not in raw_df.columns:
        raise ValueError("原始数据中也没有CVD列，无法补齐！")
    # 默认按索引一一对应
    if len(raw_df) != len(df):
        raise ValueError("原始数据与结果数据行数不一致，无法直接补齐CVD列！")
    df['CVD'] = raw_df['CVD'].values
    print("已从原始数据补齐CVD列")

# 只保留模型需要的特征，顺序严格一致，只保留 feature_names_in_ 和 df.columns 的交集
final_features = [col for col in feature_names if col in df.columns]
X = df[final_features]
print(f"最终用于建模的特征列: {final_features}")

# 检查特征数和模型参数是否一致
print(f"X.shape: {X.shape}, model.coef_.shape: {getattr(model, 'coef_', None).shape if hasattr(model, 'coef_') else '无'}")
if hasattr(model, 'coef_') and X.shape[1] != model.coef_.shape[-1]:
    raise ValueError(f"特征数量不匹配: X.shape[1]={X.shape[1]}, model.coef_.shape[-1]={model.coef_.shape[-1]}")

# SHAP 可解释性分析，强制用 LinearExplainer
explainer = shap.LinearExplainer(model, X)
shap_values = explainer(X)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
fig_path = os.path.join(figures_dir, 'pycaret_ridgeclassifier_model_shap_summary.png')
plt.savefig(fig_path, bbox_inches='tight')
plt.close()
print(f"SHAP图已保存: {fig_path}")

# 画ROC曲线
y_true = df['CVD']
if 'Score_1' in df.columns:
    y_score = df['Score_1']
else:
    # RidgeClassifier没有predict_proba，假设有决策分数列
    if 'prediction_score' in df.columns:
        y_score = df['prediction_score']
    else:
        # 退化为预测标签
        y_score = model.decision_function(X)

# 二分类标签化
classes = np.unique(y_true)
y_true_bin = label_binarize(y_true, classes=classes).ravel()
if len(classes) == 2:
    # 只处理二分类
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    roc_auc = auc(fpr, tpr)
    # micro
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin, y_score)
    auc_micro = auc(fpr_micro, tpr_micro)
    # macro
    auc_macro = roc_auc
    # 正类/负类
    fpr_pos, tpr_pos, _ = fpr, tpr, _
    auc_pos = roc_auc
    fpr_neg, tpr_neg, _ = roc_curve(1-y_true_bin, -y_score)
    auc_neg = auc(fpr_neg, tpr_neg)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_micro, tpr_micro, label=f'micro-average ROC (AUC = {auc_micro:.2f})', linestyle='--')
    plt.plot(fpr, tpr, label=f'positive ROC (AUC = {auc_pos:.2f})')
    plt.plot(fpr_neg, tpr_neg, label=f'negative ROC (AUC = {auc_neg:.2f})', linestyle=':')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    roc_fig_path = os.path.join(figures_dir, 'pycaret_ridgeclassifier_model_roc.png')
    plt.savefig(roc_fig_path, bbox_inches='tight')
    plt.close()
    print(f"ROC曲线图已保存: {roc_fig_path}")
else:
    print("当前只支持二分类的ROC曲线绘制。")
