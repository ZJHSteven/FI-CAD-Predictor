"""命令行入口：生成中文建模报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .evaluate import resolve_run_dir


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """把 DataFrame 转成不依赖 tabulate 的 Markdown 表格。

    pandas 的 `to_markdown()` 需要额外安装 `tabulate`。
    为了让报告入口在最小环境里也稳定可用，这里手写一个简单表格渲染器。
    """

    if frame.empty:
        return "（无结果）"
    text_frame = frame.fillna("").astype(str)
    headers = list(text_frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in text_frame.iterrows():
        values = [row[column].replace("|", "\\|") for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="生成 FI-CAD 第一版中文建模报告。")
    parser.add_argument("--config", default="configs/modeling.yaml", help="建模配置 YAML 路径。")
    parser.add_argument("--run", default="latest", help="run 目录或 latest。")
    return parser


def main() -> int:
    """生成 Markdown 报告。"""

    args = build_argument_parser().parse_args()
    config = load_config(args.config)
    run_dir = resolve_run_dir(args.run, config)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(manifest["tables"]["metrics"])
    report_path = run_dir / "reports" / "modeling_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    status_rank = {"ok": 0, "warning": 1, "failed": 2}
    ranked = metrics.assign(_status_rank=metrics["status"].map(status_rank).fillna(9))
    best = ranked.sort_values(["_status_rank", "roc_auc"], ascending=[True, False], na_position="last").head(1)
    best_model = best.iloc[0]["model"] if not best.empty else "无"
    lines = [
        "# FI-CAD-Predictor 第一版建模报告",
        "",
        "## 研究口径",
        "- 预测任务：2011 基线 FI 与协变量预测 2013/2015/2018/2020 新发心脏病相关事件。",
        "- 终点名称：心脏病相关事件，不把 CHARLS broader heart problems 强写成纯 CAD。",
        "- 数据泄露控制：先按 ID 划分 train/valid/test，再在模型 Pipeline 内 fit 缺失填补、标准化和分类器。",
        "",
        "## 样本概况",
        f"- 样本数：{manifest['sample_count']}",
        f"- 阳性率：{manifest['positive_rate']:.4f}",
        f"- 特征数：{manifest['feature_count']}",
        f"- Git commit：{manifest['git_commit']}",
        "",
        "## 模型结果",
        dataframe_to_markdown(metrics),
        "",
        "## 数据诊断图",
        f"- FI 风险关系图：`{manifest.get('figures', {}).get('fi_risk_curve', '未生成')}`",
        f"- 相关性热图：`{manifest.get('figures', {}).get('correlation_heatmap', '未生成')}`",
        "",
        "## 当前最佳模型",
        f"- 按当前指标表排序的候选最佳模型：`{best_model}`。",
        "- 如果某模型状态为 failed 或 warning，应优先查看 message 字段和对应混淆矩阵，不要只看 AUC。",
        "- 第一版若整体处于 warning，说明模型已跑通但还不适合作为最终论文结果，需要继续改进特征工程、终点口径复核或阈值策略。",
        "",
        "## 论文素材位置",
        f"- 指标表：`{manifest['tables']['metrics']}`",
        f"- 阈值表：`{manifest['tables']['thresholds']}`",
        f"- 模型和图表目录：`{run_dir}`",
        "",
        "## 仍需复核",
        "- 2015 血检 RAW SHA1 仍未完全闭合；第一版主模型默认不使用血检增强特征。",
        "- 第一版是固定时间窗二分类；分段风险和生存分析属于第二阶段。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已生成：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
