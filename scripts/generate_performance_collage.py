"""生成 FI-CAD 模型性能综合拼图。

这个脚本的职责非常单一：把一次训练 run 已经产出的指标表和图表底层 CSV
重新组织成一张论文汇报用的综合性能图。它不会重新训练模型，也不会改动原始
数据，因此适合作为“结果展示层”的可复现入口。

输入：
- `output/runs/<run_id>/tables/model_metrics.csv`
- `output/runs/<run_id>/tables/figure_data/*_roc_data.csv`
- `output/runs/<run_id>/tables/figure_data/*_calibration_data.csv`
- `output/runs/<run_id>/tables/figure_data/*_confusion_matrix_data.csv`

输出：
- `output/runs/<run_id>/figures/performance_collage.png`
- `output/runs/<run_id>/tables/figure_data/performance_collage_sources.csv`

设计取舍：
- “全特征集”在当前配置里对应 `primary` 特征集；其它三个特征集用于第 6 张
  分组柱状图的交叉比较。
- 所有数值都来自训练阶段已保存的 CSV，避免手工读取 PNG 或重复计算造成口径漂移。
- 图内中文依赖 Windows 常见中文字体；若环境没有中文字体，Matplotlib 会回退，
  但脚本仍能生成图片。
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

# 服务器或命令行环境通常没有 GUI，因此固定使用 Agg 后端直接保存 PNG。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from src.fi_cad.config import load_config
from src.fi_cad.evaluate import resolve_run_dir


# 当前训练配置中的五种模型顺序。显式写出顺序可以保证每次出图稳定。
MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"]

# 当前训练配置中的四种特征集顺序。第一个 primary 即用户说的“全特征集”。
FEATURE_SET_ORDER = ["primary", "no_body_size_demographic", "literature_fi_minimal", "broad_fi_minimal"]

# 面向图片读者的短标签。底层 CSV 仍保留英文机器可读名称。
MODEL_LABELS = {
    "logistic_regression": "Logistic 回归",
    "random_forest": "随机森林",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}

FEATURE_SET_LABELS = {
    "primary": "全特征集",
    "no_body_size_demographic": "去体型/人口学",
    "literature_fi_minimal": "11项FI最小集",
    "broad_fi_minimal": "宽FI最小集",
}

TARGET_LABELS = {
    "heart_related_event_by_2013": "2013",
    "heart_related_event_by_2015": "2015",
    "heart_related_event_by_2018": "2018",
    "heart_related_event_by_2020": "2020",
}

MODEL_COLORS = {
    "logistic_regression": "#1f77b4",
    "random_forest": "#ff7f0e",
    "xgboost": "#2ca02c",
    "lightgbm": "#d62728",
    "catboost": "#9467bd",
}


@dataclass(frozen=True)
class CollageInputs:
    """集中保存绘图需要的路径和表格。

    字段说明：
    - run_dir: 当前 run 的根目录。
    - metrics: 模型指标总表，每一行对应一个“时间窗 x 特征集 x 模型”组合。
    - figure_data_dir: ROC、校准曲线、混淆矩阵等底层 CSV 所在目录。
    """

    run_dir: Path
    metrics: pd.DataFrame
    figure_data_dir: Path


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数。

    返回值：
    - argparse.ArgumentParser: 标准命令行解析器。

    核心逻辑：
    - 默认读取 `configs/modeling.yaml`。
    - 默认使用 `latest`，也允许显式传入某个 run 目录，便于论文定稿时锁定版本。
    - 默认文件名固定为 `performance_collage.png`，避免每次运行生成一堆重复图。
    """

    parser = argparse.ArgumentParser(description="生成 FI-CAD 性能图像综合拼图。")
    parser.add_argument("--config", default="configs/modeling.yaml", help="建模配置 YAML 路径。")
    parser.add_argument("--run", default="latest", help="run 目录或 latest。")
    parser.add_argument("--output-name", default="performance_collage.png", help="输出 PNG 文件名。")
    return parser


def configure_matplotlib() -> None:
    """配置 Matplotlib 中文字体和导出风格。

    输入：
    - 无。

    输出：
    - 无直接返回值；通过 rcParams 修改全局绘图行为。

    核心逻辑：
    - Windows 中文环境优先尝试 Microsoft YaHei。
    - `axes.unicode_minus=False` 用来避免负号在中文字体下显示成方块。
    - 统一字号和线宽，让最终拼图在缩放后仍可读。
    """

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, table_name: str) -> None:
    """检查表格是否包含必需列。

    输入：
    - frame: 要检查的 DataFrame。
    - columns: 必需列名列表。
    - table_name: 错误信息里显示的表名，便于定位坏文件。

    输出：
    - 无返回值；如果缺列就抛出 ValueError。

    核心逻辑：
    - 拼图依赖固定列名。与其让后续绘图报难懂的 KeyError，不如在入口处
      一次性告诉用户缺了哪些列。
    """

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{table_name} 缺少必需列：{missing}")


def load_collage_inputs(config_path: str, run: str) -> CollageInputs:
    """读取 run 目录、manifest 和指标表。

    输入：
    - config_path: 建模配置文件路径。
    - run: `latest` 或具体 run 目录。

    输出：
    - CollageInputs: 后续所有绘图函数需要的路径和表格。

    核心逻辑：
    - 使用项目已有的 `resolve_run_dir`，避免重新实现 latest 解析规则。
    - 优先从 `run_manifest.json` 读取指标表路径；这比硬编码路径更稳。
    - 检查 figure_data 目录是否存在，因为 ROC/校准/混淆矩阵都依赖底层 CSV。
    """

    config = load_config(config_path)
    run_dir = resolve_run_dir(run, config)
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 run_manifest.json：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics_path = Path(manifest["tables"]["metrics"])
    metrics = pd.read_csv(metrics_path)
    require_columns(
        metrics,
        [
            "target_column",
            "feature_set",
            "model",
            "roc_auc",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp",
        ],
        table_name=str(metrics_path),
    )
    figure_data_dir = run_dir / "tables" / "figure_data"
    if not figure_data_dir.exists():
        raise FileNotFoundError(f"找不到图表底层数据目录：{figure_data_dir}")
    return CollageInputs(run_dir=run_dir, metrics=metrics, figure_data_dir=figure_data_dir)


def target_sort_key(target_column: str) -> int:
    """从终点列名中提取年份，用于时间窗排序。

    输入：
    - target_column: 例如 `heart_related_event_by_2020`。

    输出：
    - int: 提取到的年份；提取失败时返回很大的数字，让异常名称排在最后。
    """

    match = re.search(r"(20\d{2})$", str(target_column))
    return int(match.group(1)) if match else 9999


def ordered_targets(metrics: pd.DataFrame) -> list[str]:
    """按年份排序并返回当前指标表真实包含的时间窗。

    输入：
    - metrics: 模型指标表。

    输出：
    - list[str]: 已排序的 target_column 列表。
    """

    return sorted(metrics["target_column"].dropna().unique().tolist(), key=target_sort_key)


def model_label(model_name: str) -> str:
    """把机器可读模型名转成图里显示的中文名。"""

    return MODEL_LABELS.get(model_name, model_name)


def target_label(target_column: str) -> str:
    """把机器可读终点名转成图里显示的年份标签。"""

    return TARGET_LABELS.get(target_column, str(target_sort_key(target_column)))


def feature_set_label(feature_set: str) -> str:
    """把机器可读特征集名转成图里显示的中文名。"""

    return FEATURE_SET_LABELS.get(feature_set, feature_set)


def read_figure_csv(inputs: CollageInputs, stem: str, suffix: str) -> pd.DataFrame:
    """读取某个模型对应的一份图表底层 CSV。

    输入：
    - inputs: 已解析的 run 输入。
    - stem: `target__feature_set__model` 形式的模型标识。
    - suffix: 文件后缀，例如 `roc_data` 或 `calibration_data`。

    输出：
    - pd.DataFrame: 读取后的底层数据。

    异常：
    - 文件不存在时抛 FileNotFoundError，避免缺图时静默生成不完整拼图。
    """

    path = inputs.figure_data_dir / f"{stem}_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到拼图所需底层 CSV：{path}")
    return pd.read_csv(path)


def add_panel_title(ax: Axes, panel: str, title: str) -> None:
    """给子图添加统一格式标题。"""

    ax.set_title(f"{panel}. {title}", loc="left", fontweight="bold", pad=10)


def draw_auc_trend(ax: Axes, inputs: CollageInputs, targets: list[str]) -> None:
    """绘制四个时间窗、五种模型的 AUC 折线图。

    输入：
    - ax: Matplotlib 子图。
    - inputs: 拼图输入。
    - targets: 已排序的时间窗列表。

    输出：
    - 直接在 ax 上绘图。

    核心逻辑：
    - 只取 `primary`，也就是当前项目配置里的全特征集。
    - 横轴是 2013/2015/2018/2020，纵轴是 ROC-AUC。
    - 每个模型一条线，展示预测时间窗拉长时区分度的变化。
    """

    add_panel_title(ax, "1", "全特征集 AUC 随预测时间窗变化")
    primary = inputs.metrics[inputs.metrics["feature_set"] == "primary"]
    x = np.arange(len(targets))
    for model_name in MODEL_ORDER:
        rows = primary[primary["model"] == model_name].set_index("target_column")
        y = [float(rows.loc[target, "roc_auc"]) for target in targets]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            color=MODEL_COLORS.get(model_name),
            label=model_label(model_name),
        )
        for x_i, y_i in zip(x, y):
            ax.text(x_i, y_i + 0.006, f"{y_i:.3f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0.70, color="#777777", linestyle="--", linewidth=1, label="0.70 警戒线")
    ax.set_xticks(x, [target_label(target) for target in targets])
    ax.set_xlabel("预测终点年份")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.50, max(0.74, primary["roc_auc"].max() + 0.04))
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.legend(ncol=2, frameon=False)


def draw_f1_heatmap(ax: Axes, inputs: CollageInputs, targets: list[str]) -> None:
    """绘制全特征集下五种模型、四个时间窗的 F1 热力图。"""

    add_panel_title(ax, "2", "全特征集 F1 分数热力图")
    primary = inputs.metrics[inputs.metrics["feature_set"] == "primary"]
    heatmap = primary.pivot(index="model", columns="target_column", values="f1").reindex(index=MODEL_ORDER, columns=targets)
    matrix = heatmap.to_numpy(dtype=float)
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(targets)), [target_label(target) for target in targets])
    ax.set_yticks(np.arange(len(MODEL_ORDER)), [model_label(model_name) for model_name in MODEL_ORDER])
    ax.set_xlabel("预测终点年份")
    ax.set_ylabel("模型")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            ax.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=8, color="#111111")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("F1")


def confusion_matrix_from_table(table: pd.DataFrame) -> np.ndarray:
    """把长表形式的混淆矩阵转成 2x2 矩阵。

    输入：
    - table: 包含 true_label、predicted_label、count 三列的 DataFrame。

    输出：
    - np.ndarray: [[TN, FP], [FN, TP]] 结构。
    """

    require_columns(table, ["true_label", "predicted_label", "count"], table_name="confusion_matrix_data")
    matrix = np.zeros((2, 2), dtype=int)
    for _, row in table.iterrows():
        true_label = int(row["true_label"])
        predicted_label = int(row["predicted_label"])
        matrix[true_label, predicted_label] = int(row["count"])
    return matrix


def draw_single_confusion_matrix(ax: Axes, matrix: np.ndarray, title: str, normalizer: Normalize, cmap: str) -> None:
    """绘制单个模型的 2x2 混淆矩阵。"""

    ax.imshow(matrix, cmap=cmap, norm=normalizer)
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xticks([0, 1], ["预测0", "预测1"])
    ax.set_yticks([0, 1], ["真实0", "真实1"])
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    for row_index in range(2):
        for col_index in range(2):
            ax.text(
                col_index,
                row_index,
                f"{labels[row_index, col_index]}\n{matrix[row_index, col_index]:,}",
                ha="center",
                va="center",
                fontsize=9,
                color="#111111",
            )


def draw_confusion_row(fig: plt.Figure, subgrid, inputs: CollageInputs) -> list[Axes]:
    """绘制 2020 终点全特征集下五种模型混淆矩阵组图。

    输入：
    - fig: 总画布。
    - subgrid: GridSpec 中给混淆矩阵行预留的区域。
    - inputs: 拼图输入。

    输出：
    - list[Axes]: 五个混淆矩阵子图的 Axes，用于后续统一色条。
    """

    axes = [fig.add_subplot(subgrid[0, index]) for index in range(len(MODEL_ORDER))]
    matrices: list[np.ndarray] = []
    for model_name in MODEL_ORDER:
        stem = f"heart_related_event_by_2020__primary__{model_name}"
        matrix_table = read_figure_csv(inputs, stem, "confusion_matrix_data")
        matrices.append(confusion_matrix_from_table(matrix_table))
    max_count = max(int(matrix.max()) for matrix in matrices)
    normalizer = Normalize(vmin=0, vmax=max_count)
    for ax, model_name, matrix in zip(axes, MODEL_ORDER, matrices):
        draw_single_confusion_matrix(ax, matrix, model_label(model_name), normalizer, "Blues")
    axes[0].text(
        -0.48,
        1.22,
        "3. 2020 终点全特征集混淆矩阵对比",
        transform=axes[0].transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    return axes


def draw_roc_overlay(ax: Axes, inputs: CollageInputs) -> None:
    """绘制 2020 终点全特征集下五种模型 ROC 曲线叠加图。"""

    add_panel_title(ax, "4", "2020 终点全特征集 ROC 曲线叠加")
    metrics = inputs.metrics[
        (inputs.metrics["target_column"] == "heart_related_event_by_2020")
        & (inputs.metrics["feature_set"] == "primary")
    ].set_index("model")
    for model_name in MODEL_ORDER:
        stem = f"heart_related_event_by_2020__primary__{model_name}"
        roc_data = read_figure_csv(inputs, stem, "roc_data")
        require_columns(roc_data, ["fpr", "tpr"], table_name=f"{stem}_roc_data")
        auc = float(metrics.loc[model_name, "roc_auc"])
        ax.plot(
            roc_data["fpr"],
            roc_data["tpr"],
            linewidth=2,
            color=MODEL_COLORS.get(model_name),
            label=f"{model_label(model_name)} AUC={auc:.3f}",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    ax.set_xlabel("假阳性率 FPR")
    ax.set_ylabel("真阳性率 TPR")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(linestyle=":", alpha=0.45)
    ax.legend(frameon=False, loc="lower right")


def draw_logistic_calibration(ax: Axes, inputs: CollageInputs, targets: list[str]) -> None:
    """绘制全特征集 Logistic 回归在四个时间窗的校准曲线。"""

    add_panel_title(ax, "5", "全特征集 Logistic 回归校准曲线")
    colors = ["#0b6e4f", "#4f6d7a", "#c1666b", "#7d4f50"]
    for target, color in zip(targets, colors):
        stem = f"{target}__primary__logistic_regression"
        calibration = read_figure_csv(inputs, stem, "calibration_data")
        require_columns(
            calibration,
            ["predicted_probability_mean", "observed_event_rate"],
            table_name=f"{stem}_calibration_data",
        )
        ax.plot(
            calibration["predicted_probability_mean"],
            calibration["observed_event_rate"],
            marker="o",
            linewidth=2,
            color=color,
            label=target_label(target),
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1, label="理想校准")
    ax.set_xlabel("平均预测概率")
    ax.set_ylabel("实际事件率")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(linestyle=":", alpha=0.45)
    ax.legend(frameon=False, ncol=2)


def draw_feature_set_grouped_auc(ax: Axes, inputs: CollageInputs) -> None:
    """绘制 2020 终点五种模型与四种特征集交叉的 AUC 分组柱状图。"""

    add_panel_title(ax, "6", "2020 终点：模型 x 特征集 AUC 分组柱状图")
    subset = inputs.metrics[inputs.metrics["target_column"] == "heart_related_event_by_2020"]
    pivot = subset.pivot(index="model", columns="feature_set", values="roc_auc").reindex(
        index=MODEL_ORDER,
        columns=FEATURE_SET_ORDER,
    )
    x = np.arange(len(MODEL_ORDER))
    width = 0.18
    offsets = (np.arange(len(FEATURE_SET_ORDER)) - (len(FEATURE_SET_ORDER) - 1) / 2) * width
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for feature_set, offset, color in zip(FEATURE_SET_ORDER, offsets, colors):
        values = pivot[feature_set].to_numpy(dtype=float)
        bars = ax.bar(x + offset, values, width=width, color=color, label=feature_set_label(feature_set), alpha=0.88)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    ax.axhline(0.70, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(x, [model_label(model_name) for model_name in MODEL_ORDER], rotation=10, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.50, max(0.76, float(pivot.max().max()) + 0.05))
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.legend(frameon=False, ncol=2)


def write_source_index(inputs: CollageInputs, output_path: Path) -> Path:
    """写出综合拼图用到的底层数据索引。

    输入：
    - inputs: 拼图输入。
    - output_path: 最终 PNG 路径。

    输出：
    - Path: sources CSV 路径。

    核心逻辑：
    - 这不是绘图必需文件，但对复核非常重要：以后看到拼图时，可以立刻知道
      每个子图对应哪些 CSV，而不是只相信图片。
    """

    rows: list[dict[str, str]] = [
        {
            "panel": "1_auc_trend",
            "source": str(inputs.run_dir / "tables" / "model_metrics.csv"),
            "filter": "feature_set == primary",
            "output": str(output_path),
        },
        {
            "panel": "2_f1_heatmap",
            "source": str(inputs.run_dir / "tables" / "model_metrics.csv"),
            "filter": "feature_set == primary",
            "output": str(output_path),
        },
        {
            "panel": "6_feature_set_grouped_auc",
            "source": str(inputs.run_dir / "tables" / "model_metrics.csv"),
            "filter": "target_column == heart_related_event_by_2020",
            "output": str(output_path),
        },
    ]
    for model_name in MODEL_ORDER:
        stem = f"heart_related_event_by_2020__primary__{model_name}"
        rows.append(
            {
                "panel": "3_confusion_matrix",
                "source": str(inputs.figure_data_dir / f"{stem}_confusion_matrix_data.csv"),
                "filter": model_name,
                "output": str(output_path),
            }
        )
        rows.append(
            {
                "panel": "4_roc_overlay",
                "source": str(inputs.figure_data_dir / f"{stem}_roc_data.csv"),
                "filter": model_name,
                "output": str(output_path),
            }
        )
    for target in ordered_targets(inputs.metrics):
        stem = f"{target}__primary__logistic_regression"
        rows.append(
            {
                "panel": "5_logistic_calibration",
                "source": str(inputs.figure_data_dir / f"{stem}_calibration_data.csv"),
                "filter": target,
                "output": str(output_path),
            }
        )
    source_path = inputs.figure_data_dir / "performance_collage_sources.csv"
    pd.DataFrame(rows).to_csv(source_path, index=False, encoding="utf-8-sig")
    return source_path


def generate_collage(inputs: CollageInputs, output_name: str) -> tuple[Path, Path]:
    """生成综合性能拼图。

    输入：
    - inputs: run 路径、指标表和 figure_data 目录。
    - output_name: 输出 PNG 文件名。

    输出：
    - tuple[Path, Path]: 图片路径和 sources CSV 路径。

    核心逻辑：
    - 使用 Matplotlib GridSpec 搭建 4 行布局。
    - 第 3 部分的 5 个混淆矩阵使用子 GridSpec 并排展示。
    - 保存前写入来源索引，保证图和数据索引同步存在。
    """

    configure_matplotlib()
    targets = ordered_targets(inputs.metrics)
    if targets != [target for target in TARGET_LABELS if target in targets]:
        # 这里不是强制错误，只是按年份排序后继续画；真实缺失会在取数时抛错。
        targets = sorted(targets, key=target_sort_key)
    if "heart_related_event_by_2020" not in targets:
        raise ValueError("指标表里缺少 2020 终点，无法生成用户指定的 2020 对比图。")

    output_path = inputs.run_dir / "figures" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(24, 26), layout="constrained")
    grid = fig.add_gridspec(4, 2, height_ratios=[1.15, 0.9, 1.15, 1.1], width_ratios=[1.15, 1.0])

    draw_auc_trend(fig.add_subplot(grid[0, 0]), inputs, targets)
    draw_f1_heatmap(fig.add_subplot(grid[0, 1]), inputs, targets)

    confusion_grid = grid[1, :].subgridspec(1, len(MODEL_ORDER), wspace=0.06)
    confusion_axes = draw_confusion_row(fig, confusion_grid, inputs)
    # 五个混淆矩阵共用一个色条，读者能直接比较数量深浅。
    first_image = confusion_axes[-1].images[0]
    colorbar = fig.colorbar(first_image, ax=confusion_axes, fraction=0.02, pad=0.015)
    colorbar.set_label("样本数")

    draw_roc_overlay(fig.add_subplot(grid[2, 0]), inputs)
    draw_logistic_calibration(fig.add_subplot(grid[2, 1]), inputs, targets)
    draw_feature_set_grouped_auc(fig.add_subplot(grid[3, :]), inputs)

    fig.suptitle(
        "FI-CAD 纵向预测模型性能综合拼图（2011 基线 -> 2013/2015/2018/2020 终点）",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    source_path = write_source_index(inputs, output_path)
    return output_path, source_path


def main() -> int:
    """命令行主入口。

    输出：
    - 0 表示成功。
    """

    args = build_argument_parser().parse_args()
    inputs = load_collage_inputs(args.config, args.run)
    image_path, source_path = generate_collage(inputs, args.output_name)
    print(f"性能拼图已生成：{image_path}")
    print(f"拼图数据来源索引：{source_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
