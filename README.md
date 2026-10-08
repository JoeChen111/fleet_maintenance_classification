# 维护需求分类：Logistic Regression 与 Random Forest

使用已有的 `train_set.csv` 和 `test_set.csv`，比较逻辑回归与随机森林。
这是在**给定的已平衡数据上进行的个人建模项目**，不要求重现论文数值。
运行程序即可生成指标表、图表、逐条预测和中文结果报告，方便整理到 PPT 或 GitHub。

如果习惯 Jupyter / VS Code，可以在已有的 Notebook 环境中打开 `quick_start.ipynb`。
它提供路径配置、运行进度和结果展示，调用同一份分析脚本。当前 Notebook 未预填任何实验结果。

## 1. 安装与运行

建议使用 Python 3.11–3.13。先在终端进入本项目文件夹，再建立独立环境。

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

将两份 CSV 放入 `data/`，执行：

```bash
python run_analysis.py --output results/run_01 --save-models
```

Windows 未激活环境时，将命令开头的 `python` 改成 `.venv\Scripts\python.exe`。

文件已经放在电脑其他位置时，直接指定路径即可，不用复制：

```bash
python run_analysis.py \
  --train "/完整路径/train_set.csv" \
  --test "/完整路径/test_set.csv" \
  --output results/run_01 \
  --save-models
```

路径包含空格或中文时保留引号。Windows 可以将上面命令写成一行。
每次运行使用一个新的输出目录，例如 `results/run_02`，避免不同设置的结果混在一起。

只想先检查流程能否执行时，可以减少树和置换次数、跳过交叉验证：

```bash
python run_analysis.py --output results/quick_check --cv-folds 0 --trees 30 --importance-repeats 2
```

正式结果建议保留默认的 300 棵树、5 折交叉验证、5 次置换。运行时间取决于电脑配置。
程序会逐折打印进度；默认使用 4 个 CPU 线程，可用 `--jobs 2` 降低资源占用。
上述设置是事先选定的项目基线，没有进行超参数搜索。

## 2. 实际执行的步骤

1. **读取并核对数据**：检查字段、标签、缺失值、重复行、两份文件之间完全相同的特征行，以及类别比例。
2. **训练集内验证**：两个模型使用相同的分层 5 折。每折单独拟合缺失值填补和类别编码，逻辑回归额外标准化数值字段。
3. **最终拟合与评估**：用全部训练数据拟合两个模型，再评估给定测试集。两者使用预先设定的 0.5 阈值。
4. **解释模型**：导出逻辑回归系数与参考类别；在固定的测试子样本上计算两种模型的置换重要性。
5. **保存结果**：生成 CSV、JSON、PNG、SVG 和中文 Markdown 报告。可选择保存含全部预处理步骤的模型。

整个流程**不会再次做 SMOTE、不会重新合并/划分两份文件、不会基于测试结果自动调参**。
重要性分析用于解释已经固定的模型；如果据此调整字段，应在训练集内部重新验证，并另寻独立测试数据。

## 3. 输入字段

标签为 `Need_Maintenance`：`0` 表示不需要维护，`1` 表示需要维护。
训练与评估文件必须同时包含两个类别。标签不能缺失。

| 类型 | 字段 |
|---|---|
| 数值 | Mileage, Reported_Issues, Vehicle_Age, Engine_Size, From last service, Service_History, Accident_History, Fuel_Efficiency, Intensity |
| 类别 | Vehicle_Model, Maintenance_History, Fuel_Type, Transmission_Type, Owner_Type, Tire_Condition |

程序保留全部 15 个字段，采用明确的字段白名单。若 CSV 多了一列索引，如 `Unnamed: 0`，请先删除该索引列。
数值特征缺失时使用当前训练部分的中位数，类别缺失时使用当前训练部分的众数。
未知类别在 One-Hot 编码后为全零，等同于该字段的参考类别；程序同时记录测试集出现的新类别。

## 4. 模型配置

| 配置 | Logistic Regression | Random Forest |
|---|---|---|
| 数值处理 | 中位数填补，StandardScaler 标准化 | 中位数填补 |
| 类别处理 | 众数填补，One-Hot，删除一个参考水平 | 相同 |
| 主要参数 | C=1.0，lbfgs，max_iter=3000 | 300 棵树，max_features=sqrt，min_samples_leaf=2，Gini，无深度上限 |
| 类别权重 | 不额外加权 | 不额外加权 |
| 分类阈值 | 默认 0.5 | 默认 0.5 |
| 随机种子 | 42 | 42 |

`--trees`、`--seed`、`--threshold` 可在运行前指定。不要为了提高测试分数而反复修改它们。
如果需要调参或选择阈值，应只在训练集内部完成；本版代码使用固定设置，便于解释与复运行。

## 5. 输出文件

```text
results/run_01/
├── REPORT.md                         # 自动生成的中文实验结果与分析
├── data_audit.json                   # 数据规模、缺失、重复、标签比例等
├── run_metadata.json                 # 输入文件哈希、版本、参数、运行时间
├── requirements_used.txt             # 本次实际使用的依赖版本
├── classification_reports.json       # 两类各自的 precision / recall / F1
├── category_reference_levels.json    # 逻辑回归类别字段的参考水平
├── test_predictions.csv              # 测试集逐条概率与分类结果
├── tables/
│   ├── test_metrics.csv              # 两模型的主要测试指标
│   ├── cv_folds.csv                  # 每折结果；cv-folds=0 时不生成
│   ├── cv_summary.csv                # 交叉验证均值与折间标准差
│   ├── logistic_regression_confusion_matrix.csv
│   ├── random_forest_confusion_matrix.csv
│   ├── logistic_coefficients.csv
│   ├── permutation_importance.csv
│   ├── train_numeric_summary.csv
│   └── test_numeric_summary.csv
├── figures/                          # 每张图都有 PNG 与 SVG
│   ├── 01_class_distribution.*
│   ├── 02_model_comparison.*
│   ├── 03_confusion_matrices.*
│   ├── 04_roc_curves.*
│   ├── 05_precision_recall_curves.*
│   ├── 06_permutation_importance.*
│   └── 07_logistic_coefficients.*
└── models/                           # 使用 --save-models 时生成
    ├── logistic_regression.joblib
    └── random_forest.joblib
```

图表使用英文标签以避免电脑缺少中文字体；报告、说明与注释使用中文。
`source_csv_row` 在普通单行记录 CSV 中对应原文件行号（表头为第 1 行）。
输出的 `requirements_used.txt` 可用于以后建立相同依赖版本的环境。

## 6. 如何解释结果

| 指标 | 对这个项目的含义 |
|---|---|
| Accuracy | 所有记录中预测正确的比例 |
| Balanced accuracy | Recall 与 Specificity 的平均值 |
| Precision | 被判断需要维护的记录中，实际需要维护的比例 |
| Recall | 实际需要维护的记录中被模型找出的比例 |
| Specificity | 实际不需要维护的记录中被正确识别的比例 |
| F1 | 正类 Precision 与 Recall 的调和平均值 |
| ROC-AUC | 用连续预测概率区分两类的能力，不依赖单个分类阈值 |
| AP | Average precision，总结不同阈值下的 precision–recall 表现 |

混淆矩阵以真实类别为行、预测类别为列，类别顺序始终为 `[0, 1]`：

| | 预测 0 | 预测 1 |
|---|---|---|
| 真实 0 | TN：正确排除 | FP：误报维护需求 |
| 真实 1 | FN：漏报维护需求 | TP：正确识别 |

比较模型时，可以说明哪个模型 AUC 更高、哪个漏报更少、哪个误报更少，并引用实际数量。
不要只凭单一指标就宣称一个模型在任何场景都更好。

置换重要性按**原始字段**计算，表示随机打乱字段后 ROC-AUC 平均下降多少。
误差条为多次打乱得到的标准差，不是置信区间。相关字段可能分摊重要性。
逻辑回归正系数表示在模型其他字段保持不变时更倾向预测为需要维护；数值字段的系数对应增加一个训练集标准差。
类别字段相对于 `category_reference_levels.json` 中的参考类别。
`odds_ratio` 是赔率比，不是概率倍数；本项目使用正则化模型，不输出显著性检验或因果结论。

## 7. 项目结论的适用范围

两份数据已经接近平衡，且此前检查发现测试集中存在疑似合成样本特征。
代码会检查 `Intensity` 与 `Mileage / Vehicle_Age` 是否一致，但不把该检查当成合成样本的确定标签，也不据此删除记录。

因此，这个项目回答的是：**两个模型在所提供的数据版本上表现如何**。
当前 Pipeline 可以避免本次新增的预处理泄漏，但不能消除之前采样可能形成的跨文件或跨折依赖。
即使交叉验证稳定，也不证明真实车队中的独立泛化效果。测试概率不能直接解释为真实车队的已校准维护风险。
标签描述维护需求，数据不支持预测未来故障时间，也没有证据证明实际降低了维修成本。

## 8. 使用保存的模型

先在分析时加上 `--save-models`。预测文件需包含相同的 15 个特征字段，可不含标签：

```bash
python predict.py \
  --model results/run_01/models/random_forest.joblib \
  --input data/new_vehicles.csv \
  --output results/new_vehicle_predictions.csv
```

该命令会沿用保存模型中的预处理与分类阈值。模型文件只应从自己或可信来源加载。

## 9. 检查程序

安装依赖后执行：

```bash
python -m unittest discover -s tests -v
```

检查包括：阈值边界和混淆矩阵方向、训练统计量不受新数据影响、新类别处理、非法标签拒绝、置换子样本保留两类，以及完整训练/报告/模型保存/再次预测流程。
检查中的数据是临时构造的数据，不能当成这个项目的实际实验结果。

## 10. 整理成 GitHub 项目与 PPT

运行后先阅读 `REPORT.md`，再从 `figures/` 中挑选图表放入 PPT。
可使用“项目目标、数据、方法流程、模型结果、误判分析、变量解释、结论与局限”七页结构。
GitHub 可放代码、README、汇总结果、图表和 PPT 导出的 PDF。
默认忽略原始 CSV、大体积模型和逐条测试预测；确认数据可公开后再决定是否分享原始数据。

技术参考：[scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)、
[置换重要性](https://scikit-learn.org/stable/modules/permutation_importance.html)。
