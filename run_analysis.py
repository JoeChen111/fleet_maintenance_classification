"""在给定的平衡数据上比较 Logistic Regression 和 Random Forest。

运行示例：python run_analysis.py --train data/train_set.csv --test data/test_set.csv
流程：数据核对 -> 训练集分层交叉验证 -> 全训练集拟合 -> 给定测试集评估 -> 图表/报告。
本项目不进行再次采样，也不使用测试集选择参数或分类阈值。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # 支持没有图形界面的环境，图表直接保存到文件。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Need_Maintenance"
NUMERIC = [
    "Mileage", "Reported_Issues", "Vehicle_Age", "Engine_Size",
    "From last service", "Service_History", "Accident_History",
    "Fuel_Efficiency", "Intensity",
]
CATEGORICAL = [
    "Vehicle_Model", "Maintenance_History", "Fuel_Type",
    "Transmission_Type", "Owner_Type", "Tire_Condition",
]
FEATURES = NUMERIC + CATEGORICAL
METRICS = [
    "accuracy", "balanced_accuracy", "precision", "recall",
    "specificity", "f1", "roc_auc", "average_precision",
]
COLORS = {"Logistic Regression": "#2563EB", "Random Forest": "#0F766E"}
SLUGS = {"Logistic Regression": "logistic_regression", "Random Forest": "random_forest"}


def load_data(path: Path, require_target: bool = True) -> pd.DataFrame:
    """严格检查字段；缺失特征留给训练管道填补，缺失/非法标签直接报错。"""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = [x.strip() for x in next(csv.reader(stream), [])]
    if len(header) != len(set(header)):
        raise ValueError(f"{path.name}: 字段名有重复。")
    data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    data.columns = data.columns.str.strip()
    required = FEATURES + ([TARGET] if require_target else [])
    missing = sorted(set(required) - set(data.columns))
    extra = sorted(set(data.columns) - set(FEATURES + [TARGET]))
    if missing or extra:
        raise ValueError(f"{path.name}: 缺少字段 {missing}；额外字段 {extra}。")
    if data.empty:
        raise ValueError(f"{path.name}: 数据为空。")
    for column in NUMERIC:
        data[column] = pd.to_numeric(data[column], errors="raise").astype(float)
        if np.isinf(data[column].to_numpy()).any():
            raise ValueError(f"{path.name}: {column} 存在无穷值。")
    for column in CATEGORICAL:
        data[column] = data[column].map(
            lambda x: str(x).strip() if pd.notna(x) and str(x).strip() else np.nan
        ).astype(object)
    if require_target:
        target = pd.to_numeric(data[TARGET], errors="raise")
        if target.isna().any() or not target.isin([0, 1]).all():
            raise ValueError(f"{path.name}: {TARGET} 必须全部为 0 或 1，不能缺失。")
        data[TARGET] = target.astype(int)
        if data[TARGET].nunique() != 2:
            raise ValueError(f"{path.name}: 评估数据必须同时包含 0、1 两类。")
    return data


def make_models(seed: int, trees: int, jobs: int) -> dict[str, Pipeline]:
    """所有填补、编码和缩放都在 Pipeline 内，只从当前训练数据学习。"""
    def preprocessing(scale: bool) -> ColumnTransformer:
        numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if scale:
            numeric_steps.append(("scale", StandardScaler()))
        categorical = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ])
        return ColumnTransformer([
            ("numeric", Pipeline(numeric_steps), NUMERIC),
            ("categorical", categorical, CATEGORICAL),
        ], remainder="drop")

    return {
        "Logistic Regression": Pipeline([
            ("preprocess", preprocessing(scale=True)),
            ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000,
                                         random_state=seed)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", preprocessing(scale=False)),
            ("model", RandomForestClassifier(
                n_estimators=trees, max_features="sqrt", min_samples_leaf=2,
                max_depth=None, criterion="gini", class_weight=None,
                random_state=seed, n_jobs=jobs,
            )),
        ]),
    }


def positive_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    index = list(model.classes_).index(1)
    return model.predict_proba(X)[:, index]


def evaluate(y: pd.Series | np.ndarray, probability: np.ndarray,
             threshold: float = 0.5) -> tuple[dict, np.ndarray, np.ndarray]:
    """1=需要维护；FN=漏报维护需求；FP=不需要维护却被判定需要维护。"""
    prediction = (probability >= threshold).astype(int)
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in matrix.ravel())
    scores = {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "specificity": tn / (tn + fp),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    return scores, prediction, matrix


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def dataset_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """描述已提供的数据；派生字段不一致是诊断信息，不用于剔除或重建样本。"""
    result = {}
    for name, data in [("train", train), ("test", test)]:
        valid = data[["Mileage", "Vehicle_Age", "Intensity"]].notna().all(axis=1) & data.Vehicle_Age.ne(0)
        matches = np.isclose(
            data.loc[valid, "Intensity"],
            data.loc[valid, "Mileage"] / data.loc[valid, "Vehicle_Age"],
            rtol=1e-9, atol=1e-6,
        )
        counts = data[TARGET].value_counts().sort_index()
        result[name] = {
            "rows": len(data), "predictors": len(FEATURES),
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "positive_rate": float(data[TARGET].mean()),
            "majority_class_accuracy_reference": float(counts.max() / len(data)),
            "missing_values": {k: int(v) for k, v in data.isna().sum().items()},
            "duplicate_full_rows": int(data.duplicated().sum()),
            "intensity_check_valid_rows": int(valid.sum()),
            "intensity_not_equal_to_mileage_divided_by_age": int((~matches).sum()),
        }
    train_hash = pd.util.hash_pandas_object(train[FEATURES], index=False)
    test_hash = pd.util.hash_pandas_object(test[FEATURES], index=False)
    result["test_rows_with_identical_training_features"] = int(test_hash.isin(train_hash).sum())
    result["unseen_test_categories"] = {
        column: sorted(set(test[column].dropna()) - set(train[column].dropna()))
        for column in CATEGORICAL
    }
    result["scope"] = (
        "仅评估提供的已平衡数据。没有再次采样。此前采样造成的依赖可能跨越文件或交叉验证折；"
        "当前 Pipeline 只能防止本次新增的预处理泄漏，不能消除已有依赖。"
        "测试指标不代表原始车队分布下的表现，输出概率也不视为已校准的真实维护风险。"
    )
    return result


def cross_validate_models(models: dict, X: pd.DataFrame, y: pd.Series,
                          folds: int, seed: int, threshold: float) -> pd.DataFrame:
    # 两个模型使用完全相同的折；每折 clone 保证预处理重新拟合。
    splits = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y))
    records = []
    for name, template in models.items():
        for fold, (training, validation) in enumerate(splits, start=1):
            started = time.perf_counter()
            pipeline = clone(template)
            pipeline.fit(X.iloc[training], y.iloc[training])
            probability = positive_probability(pipeline, X.iloc[validation])
            scores, _, _ = evaluate(y.iloc[validation], probability, threshold)
            records.append({"model": name, "fold": fold, **scores,
                            "elapsed_seconds": time.perf_counter() - started})
            print(f"[CV] {name}, fold {fold}/{folds}, AUC={scores['roc_auc']:.4f}", flush=True)
    return pd.DataFrame(records)


def coefficient_table(model: Pipeline) -> tuple[pd.DataFrame, dict]:
    preprocess = model.named_steps["preprocess"]
    encoder = preprocess.named_transformers_["categorical"].named_steps["encode"]
    names = preprocess.get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]
    units = ["每增加 1 个训练集标准差" for _ in NUMERIC]
    references = {}
    for column, categories in zip(CATEGORICAL, encoder.categories_):
        references[column] = str(categories[0])
        units.extend([f"相对于 {column}={categories[0]}" for _ in categories[1:]])
    table = pd.DataFrame({
        "feature": names, "coefficient": coefficients,
        "odds_ratio": np.exp(coefficients), "comparison": units,
    }).sort_values("coefficient", key=lambda x: x.abs(), ascending=False)
    # 系数来自带 L2 正则化的分类器，不报告 p 值或因果结论。
    return table, references


def importance_sample_indices(y: pd.Series, size: int, seed: int) -> np.ndarray:
    """按比例抽取固定子样本，确保包含两个类别，允许只剩一个未选中的样本。"""
    size = min(size, len(y))
    if size == len(y):
        return np.arange(len(y))
    values = y.to_numpy()
    zero_indices, one_indices = np.flatnonzero(values == 0), np.flatnonzero(values == 1)
    n_zero = int(np.clip(round(size * len(zero_indices) / len(y)),
                         max(1, size - len(one_indices)), min(len(zero_indices), size - 1)))
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([rng.choice(zero_indices, n_zero, replace=False),
                             rng.choice(one_indices, size - n_zero, replace=False)])
    rng.shuffle(chosen)
    return chosen


def save_figure(fig: plt.Figure, directory: Path, name: str) -> None:
    fig.savefig(directory / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_results(train: pd.DataFrame, test: pd.DataFrame, scores: pd.DataFrame,
                 predictions: dict, matrices: dict, importance: pd.DataFrame,
                 coefficients: pd.DataFrame, directory: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, (name, data) in zip(axes, [("Training data", train), ("Supplied test data", test)]):
        counts = data[TARGET].value_counts().reindex([0, 1])
        bars = ax.bar(["0: No maintenance", "1: Maintenance"], counts, color=["#94A3B8", "#2563EB"])
        ax.bar_label(bars, fmt="%.0f", padding=3)
        ax.set(title=name, ylabel="Rows", ylim=(0, counts.max() * 1.18))
    save_figure(fig, directory, "01_class_distribution")

    selected = ["roc_auc", "balanced_accuracy", "precision", "recall", "specificity", "f1"]
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    x = np.arange(len(selected))
    for i, (name, row) in enumerate(scores.iterrows()):
        bars = ax.bar(x + (i - .5) * .36, row[selected].astype(float), width=.36,
                      label=name, color=COLORS[name])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set(xticks=x, xticklabels=[s.replace("_", " ").title() for s in selected],
           ylim=(0, 1.13), title="Model comparison on supplied balanced test data", ylabel="Score")
    ax.legend(loc="upper center", ncol=2, frameon=False)
    save_figure(fig, directory, "02_model_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, (name, matrix) in zip(axes, matrices.items()):
        ConfusionMatrixDisplay(matrix, display_labels=["0: No need", "1: Need"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name)
    fig.suptitle("Supplied balanced test data: rows = actual, columns = predicted")
    save_figure(fig, directory, "03_confusion_matrices")

    fig, ax = plt.subplots(figsize=(6.5, 5.3), layout="constrained")
    for name, probability in predictions.items():
        fpr, tpr, _ = roc_curve(test[TARGET], probability)
        ax.plot(fpr, tpr, color=COLORS[name], label=f"{name}: AUC = {scores.loc[name, 'roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#94A3B8")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
           title="ROC curves — supplied balanced test data")
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, directory, "04_roc_curves")

    fig, ax = plt.subplots(figsize=(6.5, 5.3), layout="constrained")
    for name, probability in predictions.items():
        precision, recall, _ = precision_recall_curve(test[TARGET], probability)
        ax.plot(recall, precision, color=COLORS[name],
                label=f"{name}: AP = {scores.loc[name, 'average_precision']:.3f}")
    ax.axhline(test[TARGET].mean(), color="#94A3B8", linestyle="--", label="Positive-class prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curves (AP = average precision)")
    ax.legend(loc="lower left", frameon=False)
    save_figure(fig, directory, "05_precision_recall_curves")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, name in zip(axes, predictions):
        top = importance[importance.model.eq(name)].nlargest(10, "importance_mean").sort_values("importance_mean")
        ax.barh(top.feature, top.importance_mean, xerr=top.importance_std, color=COLORS[name], alpha=.85)
        ax.axvline(0, linewidth=.7, color="#64748B")
        ax.set(title=name, xlabel="Decrease in ROC-AUC after shuffling")
    fig.suptitle("Permutation importance on a fixed test sample; bars show mean ± SD")
    save_figure(fig, directory, "06_permutation_importance")

    top = coefficients.head(12).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    labels = top.feature.str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False)
    ax.barh(labels, top.coefficient, color=np.where(top.coefficient.ge(0), "#2563EB", "#94A3B8"))
    ax.axvline(0, linewidth=.8, color="#475569")
    ax.set(xlabel="Coefficient (numeric fields standardized; categorical fields vs reference)",
           title="Logistic regression: 12 largest absolute coefficients")
    save_figure(fig, directory, "07_logistic_coefficients")


def write_report(output: Path, scores: pd.DataFrame, audit: dict,
                 cv_summary: pd.DataFrame | None, importance: pd.DataFrame,
                 threshold: float) -> None:
    lines = ["# 维护需求分类模型比较", "", "本报告由代码运行后自动生成，使用给定的已平衡数据。", "",
             f"训练集：{audit['train']['rows']:,} 行；测试集：{audit['test']['rows']:,} 行。",
             f"需要维护的比例：训练集 {audit['train']['positive_rate']:.2%}；测试集 {audit['test']['positive_rate']:.2%}。",
             f"正类定义为需要维护（1），预先设定的分类阈值为 {threshold:g}。", "",
             "## 测试集结果", "",
             "| 模型 | Accuracy | Balanced accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | AP |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in scores.iterrows():
        lines.append("| " + name + " | " + " | ".join(f"{row[k]:.4f}" for k in METRICS) + " |")
    lines.extend(["", "AP 是 average precision，采用概率计算；它不是将 PR 曲线作梯形积分得到的面积。", ""])
    for name, row in scores.iterrows():
        lines.append(f"- {name}：漏报维护需求 FN={int(row.fn):,}；误报维护需求 FP={int(row.fp):,}。")
    delta = scores.loc["Random Forest", "roc_auc"] - scores.loc["Logistic Regression", "roc_auc"]
    lines.extend(["", f"随机森林减去逻辑回归的测试 ROC-AUC 差值为 {delta:+.4f}。这是描述性差异，不是显著性检验。", ""])
    if cv_summary is not None:
        lines.extend(["## 训练集内交叉验证", "", "| 模型 | ROC-AUC 均值 | 折间标准差 | F1 均值 | 折间标准差 |",
                      "|---|---:|---:|---:|---:|"])
        for _, row in cv_summary.iterrows():
            lines.append(f"| {row['model']} | {row.roc_auc_mean:.4f} | {row.roc_auc_std:.4f} | {row.f1_mean:.4f} | {row.f1_std:.4f} |")
        lines.extend(["", "折间标准差仅描述本次划分的波动，不是泛化性能的置信区间。", ""])
    lines.extend(["## 变量解释", ""])
    for name in scores.index:
        top = importance[importance.model.eq(name)].nlargest(5, "importance_mean")
        lines.append(f"- {name} 的前五个置换重要性字段：" + "、".join(top.feature) + "。")
    lines.extend(["", "置换重要性使用同一测试子样本，仅用于解释已经固定的模型；没有据此重选特征或重新训练。",
                  "相关特征可能分摊重要性。负的重要性值表示打乱该字段后本次样本上的 AUC 反而略有提高。",
                  "逻辑回归系数反映条件关联，不能解释为因果；odds ratio 是赔率比，不是概率倍数。", "",
                  "## 图表", "", "![模型比较](figures/02_model_comparison.png)", "",
                  "![混淆矩阵](figures/03_confusion_matrices.png)", "",
                  "![ROC 曲线](figures/04_roc_curves.png)", "",
                  "![变量重要性](figures/06_permutation_importance.png)", "",
                  "## 适用范围", "", audit["scope"], "",
                  f"检测到 {audit['test_rows_with_identical_training_features']:,} 行测试数据的全部特征与某行训练数据完全相同；"
                  "数量为零也不能排除近邻合成样本造成的依赖。", "",
                  "该任务判断当前记录对应的维护需求，不能据此宣称提前预测了未来故障时间或证明降低了维护成本。",
                  "类别平衡不证明样本相互独立；训练集内交叉验证也不能修复原先可能存在的采样依赖。", "",
                  "## 适合用于 PPT 的内容", "",
                  "1. 项目目标与两个模型。", "2. 数据规模、标签比例和已有数据限制。",
                  "3. 编码、标准化、交叉验证和测试流程。", "4. 模型指标与 ROC 曲线。",
                  "5. 混淆矩阵：漏报与误报。", "6. 变量重要性与逻辑回归系数。",
                  "7. 结论、适用范围和后续改进。", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=Path("data/train_set.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test_set.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"), help="使用新的或空的目录，避免混合不同运行的结果")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5, help="默认 5；0 表示跳过交叉验证")
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=4, help="随机森林并行线程数，-1 表示使用全部可用 CPU")
    parser.add_argument("--threshold", type=float, default=0.5, help="运行前指定；不得根据测试结果反复调整")
    parser.add_argument("--importance-samples", type=int, default=2000)
    parser.add_argument("--importance-repeats", type=int, default=5)
    parser.add_argument("--save-models", action="store_true", help="保存完整预处理和模型，供 predict.py 使用")
    args = parser.parse_args()
    if args.cv_folds not in [0] and args.cv_folds < 2:
        parser.error("--cv-folds 必须为 0 或至少为 2。")
    if args.trees < 1 or args.importance_repeats < 1 or args.importance_samples < 4:
        parser.error("树数量和置换重复次数须为正数，重要性样本数须至少为 4。")
    if args.jobs == 0 or args.jobs < -1 or not 0 < args.threshold < 1:
        parser.error("--jobs 必须为 -1 或正数；--threshold 必须介于 0 和 1 之间。")
    return args


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("error", category=ConvergenceWarning)
    train, test = load_data(args.train), load_data(args.test)
    all_missing = [c for c in FEATURES if train[c].isna().all()]
    if all_missing:
        raise ValueError(f"训练集存在整列缺失：{all_missing}")
    if args.cv_folds and train[TARGET].value_counts().min() < args.cv_folds:
        raise ValueError("训练集最少类别的数量小于交叉验证折数。")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"输出目录非空：{args.output}。请用 --output 指定一个新的目录。")
    args.output.mkdir(parents=True, exist_ok=True)
    tables, figures = args.output / "tables", args.output / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    audit = dataset_audit(train, test)
    write_json(args.output / "data_audit.json", audit)
    if audit["test_rows_with_identical_training_features"]:
        warnings.warn("训练集与测试集存在完全相同的特征行，独立性受影响；详见 data_audit.json。")
    print(f"数据：train={len(train):,}, test={len(test):,}；直接使用已有划分，不做重采样。", flush=True)
    for split, data in [("train", train), ("test", test)]:
        data[NUMERIC].describe().T.to_csv(tables / f"{split}_numeric_summary.csv", encoding="utf-8-sig")
    models = make_models(args.seed, args.trees, args.jobs)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    cv_summary = None
    if args.cv_folds:
        folds = cross_validate_models(models, X_train, y_train, args.cv_folds, args.seed, args.threshold)
        folds.to_csv(tables / "cv_folds.csv", index=False, encoding="utf-8-sig")
        grouped = folds.groupby("model")[METRICS].agg(["mean", "std"])
        grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
        cv_summary = grouped.reset_index()
        cv_summary.to_csv(tables / "cv_summary.csv", index=False, encoding="utf-8-sig")
    records, probabilities, matrices, reports, fitted_models = [], {}, {}, {}, {}
    prediction_table = pd.DataFrame({"source_csv_row": np.arange(len(test)) + 2, "actual": y_test})
    for name, template in models.items():
        print(f"[FIT] 正在拟合 {name} 并评估给定测试集…", flush=True)
        started = time.perf_counter()
        model = clone(template).fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        probability = positive_probability(model, X_test)
        scores, prediction, matrix = evaluate(y_test, probability, args.threshold)
        records.append({"model": name, **scores, "threshold": args.threshold, "fit_seconds": fit_seconds})
        probabilities[name], matrices[name], fitted_models[name] = probability, matrix, model
        prediction_table[f"{SLUGS[name]}_probability"] = probability
        prediction_table[f"{SLUGS[name]}_prediction"] = prediction
        reports[name] = classification_report(y_test, prediction, labels=[0, 1],
            target_names=["no_maintenance", "needs_maintenance"], output_dict=True, zero_division=0)
        pd.DataFrame(matrix, index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"]).to_csv(
            tables / f"{SLUGS[name]}_confusion_matrix.csv", encoding="utf-8-sig")
        if args.save_models:
            model_dir = args.output / "models"
            model_dir.mkdir(exist_ok=True)
            joblib.dump({"pipeline": model, "threshold": args.threshold, "features": FEATURES,
                         "name": name, "positive_class": 1}, model_dir / f"{SLUGS[name]}.joblib", compress=3)
    scores = pd.DataFrame(records).set_index("model")
    scores.to_csv(tables / "test_metrics.csv", encoding="utf-8-sig")
    prediction_table.to_csv(args.output / "test_predictions.csv", index=False, encoding="utf-8-sig")
    write_json(args.output / "classification_reports.json", reports)
    print("[EXPLAIN] 计算两个固定模型的变量重要性…", flush=True)
    # 子样本近似保留测试集类别比例，两个模型使用同一组行。
    indices = importance_sample_indices(y_test, args.importance_samples, args.seed)
    importance_tables = []
    for name, model in fitted_models.items():
        result = permutation_importance(model, X_test.iloc[indices], y_test.iloc[indices],
            scoring="roc_auc", n_repeats=args.importance_repeats, random_state=args.seed, n_jobs=1)
        importance_tables.append(pd.DataFrame({"model": name, "feature": FEATURES,
            "importance_mean": result.importances_mean, "importance_std": result.importances_std}))
    importance = pd.concat(importance_tables, ignore_index=True)
    importance.to_csv(tables / "permutation_importance.csv", index=False, encoding="utf-8-sig")
    coefficients, references = coefficient_table(fitted_models["Logistic Regression"])
    coefficients.to_csv(tables / "logistic_coefficients.csv", index=False, encoding="utf-8-sig")
    write_json(args.output / "category_reference_levels.json", references)
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "matplotlib", "joblib", "scipy", "threadpoolctl"]}
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "versions": versions, "seed": args.seed, "cv_folds": args.cv_folds,
        "threshold": args.threshold, "trees": args.trees, "jobs": args.jobs,
        "importance_sample_size": len(indices), "importance_repeats": args.importance_repeats,
        "features": FEATURES, "target": TARGET,
        "model_parameters": {name: model.named_steps["model"].get_params() for name, model in fitted_models.items()},
        "input_files": {label: {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for label, path in [("train", args.train), ("test", args.test)]},
    }
    write_json(args.output / "run_metadata.json", metadata)
    (args.output / "requirements_used.txt").write_text(
        "\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n", encoding="utf-8")
    plot_results(train, test, scores, probabilities, matrices, importance, coefficients, figures)
    write_report(args.output, scores, audit, cv_summary, importance, args.threshold)
    print("\n" + scores[METRICS].round(4).to_string())
    print(f"\n完成。结果位置：{args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
