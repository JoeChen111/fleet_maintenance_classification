"""Training-only EDA for portfolio slides 6–8.
Run: python run_eda.py
Uses pandas, numpy and matplotlib from the project's requirements.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'fleet_eda_matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

ROOT = Path(__file__).resolve().parent
TARGET = 'Need_Maintenance'
NAVY, TEAL, GRAY = '#12344F', '#318C87', '#CBD5DF'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'axes.labelcolor': NAVY,
                     'savefig.facecolor': 'white'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=ROOT/'data/train_set.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'results/eda_train')
    args = parser.parse_args()
    d = pd.read_csv(args.train)
    required = [TARGET, 'Vehicle_Age', 'Mileage', 'Reported_Issues', 'Maintenance_History', 'Service_History']
    if any(c not in d for c in required):
        raise ValueError(f'Required columns: {required}')
    if d.empty or d[TARGET].isna().any() or not set(d[TARGET].unique()).issubset({0, 1}):
        raise ValueError('Expected non-empty data and complete binary target (0/1).')
    out = args.output
    figs, tables = out/'figures', out/'tables'
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    def save_table(df, name):
        df.to_csv(tables/f'{name}.csv', index=False, encoding='utf-8-sig', float_format='%.8f')

    def save_fig(fig, name):
        fig.savefig(figs/f'{name}.png', dpi=220, bbox_inches='tight')
        fig.savefig(figs/f'{name}.svg', bbox_inches='tight')
        plt.close(fig)

    def grouped(col, order=None):
        # Retain missing groups if future input files contain missing predictors.
        x = d.groupby(col, dropna=False, observed=True)[TARGET].agg(sample_count='size', needs_maintenance_count='sum')
        if order and set(x.index) == set(order):
            x = x.reindex(order)
        x['no_maintenance_count'] = x.sample_count - x.needs_maintenance_count
        x['maintenance_rate'] = x.needs_maintenance_count/x.sample_count
        x['maintenance_rate_pct'] = 100*x.maintenance_rate
        assert x.sample_count.sum() == len(d)
        assert x.needs_maintenance_count.sum() == d[TARGET].sum()
        return x.reset_index()

    classes = d[TARGET].value_counts().reindex([0, 1], fill_value=0).rename_axis(TARGET).reset_index(name='sample_count')
    classes['share_pct'] = classes.sample_count/len(d)*100
    save_table(classes, '01_class_distribution')
    numeric = d.drop(columns=TARGET).select_dtypes(include=np.number)
    summary = numeric.describe().T.rename_axis('variable').reset_index()
    save_table(summary, '02_numeric_summary')
    age = d.groupby('Vehicle_Age', dropna=False).size().rename('sample_count').reset_index()
    save_table(age, '03_vehicle_age_distribution')
    # Fixed-width bins cover the full observed range. Last bin includes its upper edge.
    mileage = d.Mileage.dropna()
    edges = np.linspace(mileage.min(), mileage.max(), 11)
    hist, edges = np.histogram(mileage, bins=edges)
    bins = pd.DataFrame({'bin_left': edges[:-1], 'bin_right': edges[1:], 'sample_count': hist,
                         'interval': ['[left, right)']*9+['[left, right]']})
    assert hist.sum() == mileage.size
    save_table(bins, '04_mileage_histogram')
    audit = pd.DataFrame({'variable':d.columns, 'missing_count': d.isna().sum().values,
                          'unique_values': d.nunique(dropna=True).values, 'dtype': d.dtypes.astype(str).values})
    save_table(audit, '00_column_audit')

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.9))
    ax[0].bar(['0: Not required', '1: Required'], classes.sample_count, color=[GRAY, TEAL], width=.6)
    for i, row in classes.iterrows():
        ax[0].text(i, row.sample_count+900, f'{row.sample_count:,.0f}\n({row.share_pct:.2f}%)', ha='center', fontsize=11)
    ax[0].set_ylim(0, classes.sample_count.max()*1.28)
    ax[0].set_title('Maintenance labels')
    ax[0].set_ylabel('Number of records')
    ax[1].bar(age.Vehicle_Age, age.sample_count, color=NAVY, width=.7)
    ax[1].set_title('Vehicle age')
    ax[1].set_xlabel('Vehicle_Age (recorded values)')
    ax[1].set_xticks(age.Vehicle_Age)
    ax[2].bar(edges[:-1], hist, width=np.diff(edges)*.96, align='edge', color=TEAL)
    ax[2].set_title('Mileage')
    ax[2].set_xlabel('Mileage (recorded values)')
    ax[2].xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v/1000:.0f}k'))
    for a in ax:
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:,.0f}'))
        a.grid(axis='y', alpha=.15)
        a.set_axisbelow(True)
    fig.suptitle(f'Training data profile | {len(d):,} records', fontsize=19, color=NAVY, fontweight='bold')
    fig.text(.5, .015, 'Supplied balanced training data only. The class share is not a real-fleet prevalence estimate.', ha='center', fontsize=10, color='#596675')
    fig.tight_layout(rect=[0,.055,1,.91])
    save_fig(fig, '06_eda_data_profile')

    def rate_chart(col, frame, title, filename):
        fig, ax = plt.subplots(figsize=(10, 5.6))
        positions = np.arange(len(frame))
        ax.bar(positions, frame.maintenance_rate, color=TEAL, width=.62)
        overall = d[TARGET].mean()
        ax.axhline(overall, color=NAVY, ls='--', lw=1.5, label=f'Training-set average: {overall:.1%}')
        for i, row in frame.iterrows():
            ax.text(i, row.maintenance_rate+.025, f'{row.maintenance_rate:.1%}\nn={int(row.sample_count):,}', ha='center', fontsize=11, zorder=5, bbox=dict(facecolor='white', edgecolor='none', alpha=.93, pad=1.5))
        ax.set_xticks(positions, frame[col].astype(str))
        ax.set_ylim(0, 1.20)
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel('Share with Need_Maintenance = 1')
        ax.set_xlabel(col)
        ax.set_title(title, fontsize=18, color=NAVY, pad=15)
        ax.legend(loc='upper left', frameon=False, fontsize=10)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
        fig.text(.5,.015,'Training data only | Descriptive association, not causation or calibrated real-world risk.',ha='center',fontsize=10,color='#596675')
        fig.tight_layout(rect=[0,.055,1,1])
        save_fig(fig, filename)

    issues = grouped('Reported_Issues')
    history = grouped('Maintenance_History', ['Good','Average','Poor'])
    service = grouped('Service_History')
    save_table(issues, '05_reported_issues_rates')
    save_table(history, '06_maintenance_history_rates')
    save_table(service, '07_service_history_rates')
    rate_chart('Reported_Issues', issues, 'Maintenance need by reported issue count', '07_eda_reported_issues')
    rate_chart('Maintenance_History', history, 'Maintenance need by maintenance history', '08_eda_maintenance_history')
    rate_chart('Service_History', service, 'Maintenance need by service history', '08b_optional_service_history')

    def md(df):
        # Avoid requiring the optional tabulate dependency.
        headers = list(df.columns)
        rows = ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
        for record in df.itertuples(index=False, name=None):
            rows.append('| '+' | '.join(str(v) for v in record)+' |')
        return '\n'.join(rows)

    def rate_table(frame, col):
        x=frame[[col,'sample_count','needs_maintenance_count','no_maintenance_count','maintenance_rate_pct']].copy()
        x['maintenance_rate_pct']=x.maintenance_rate_pct.map(lambda v:f'{v:.2f}%')
        x.columns=[col,'记录数','需要维护数','不需要维护数','组内维护需求比例']
        return md(x)

    issues_note = ''
    if set(issues.Reported_Issues) == set(range(6)):
        r = issues.set_index('Reported_Issues').maintenance_rate
        issues_note = f'观察：问题数量从 1 到 5 时，各组维护需求比例依次为 {r[1]:.2%}、{r[2]:.2%}、{r[3]:.2%}、{r[4]:.2%}、{r[5]:.2%}。0 个问题组为 {r[0]:.2%}，所以不能笼统声称全范围严格递增。若某组为 100%，仅说明该训练文件中的该组标签全部为 1，不证明现实必然规律，也不能单凭这一点认定数据泄漏。'
    history_note = ''
    if set(history.Maintenance_History) == {'Good','Average','Poor'}:
        r = history.set_index('Maintenance_History').maintenance_rate
        history_note = f'观察：Good、Average、Poor 三组的维护需求比例分别为 {r["Good"]:.2%}、{r["Average"]:.2%}、{r["Poor"]:.2%}。Poor 比 Good 高 {(r["Poor"]-r["Good"])*100:.2f} 个百分点。这是分组关联，没有控制其他变量，不能解释为维护历史造成了这些差异。'
    notes = f'''# EDA 结果：用于 PPT 第 6–8 页

## 分析范围和读法
- 仅使用训练集：{len(d):,} 条记录；没有读取测试集，没有再次采样，没有重新训练模型。
- 组内维护需求比例 = 该组标签为 1 的记录数 ÷ 该组全部记录数。
- 图中的 n 是每组记录数，不代表独立车辆数（数据没有唯一车辆标识可供确认）。
- 已平衡数据中的比例不等于真实车队的维护发生率；观察到关联不等于证明因果。
- 保留原始字段值；未确认单位的字段用 recorded values，不自行添加单位。

## 第 6 页：EDA — Data Profile
图：figures/06_eda_data_profile.png
- 标签 0：{int(classes.iloc[0].sample_count):,} 条，标签 1：{int(classes.iloc[1].sample_count):,} 条；两类基本各占 50%。
- Vehicle_Age 的范围为 {d.Vehicle_Age.min():g}–{d.Vehicle_Age.max():g}，中位数为 {d.Vehicle_Age.median():g}。
- Mileage 的范围为 {d.Mileage.min():,}–{d.Mileage.max():,}，中位数为 {d.Mileage.median():,.0f}。
- 可写入 PPT：The supplied training data are balanced across maintenance labels. Vehicle age and mileage cover a range of recorded values.
- 本页展示总体分布，不能据此说车龄或里程越高就越需要维护。

## 第 7 页：EDA — Reported Issues
图：figures/07_eda_reported_issues.png

{rate_table(issues,'Reported_Issues')}

读图方法：每根柱表示该问题数量组内的维护需求比例，不是该组占全部记录的比例。结合上表逐组比较，不预设比例一定单调上升。

{issues_note}

## 第 8 页：EDA — Maintenance History
图：figures/08_eda_maintenance_history.png

{rate_table(history,'Maintenance_History')}

{history_note}

## 可选补充：Service History
图：figures/08b_optional_service_history.png

{rate_table(service,'Service_History')}

## 输出文件与复跑
- figures：英文图，PNG 可直接放入 PPT；SVG 可缩放。
- tables：对应 CSV；maintenance_rate 为 0–1 小数，maintenance_rate_pct 为百分数值。
- run_eda.py：项目根目录中的完整 Python 脚本；从项目文件夹运行 `python run_eda.py`。
- 分析使用全部训练记录，不抽样、不删除离群值。缺失情况见 00_column_audit.csv。
- 本次图表与报告依据实际 CSV 计算。数据先前平衡或合成处理的影响仍需保留在项目限制中。
'''
    (out/'EDA_README_中文.md').write_text(notes, encoding='utf-8')
    metadata={'train_path':str(args.train.resolve()), 'train_sha256':hashlib.sha256(args.train.read_bytes()).hexdigest(),
              'rows':len(d), 'missing_cells':int(d.isna().sum().sum()), 'duplicate_full_rows':int(d.duplicated().sum()),
              'python':sys.version, 'pandas':pd.__version__, 'numpy':np.__version__, 'matplotlib':matplotlib.__version__,
              'scope':'All supplied training records only; no resampling, no test data, no model fitting.'}
    (out/'eda_metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Reported issues:\n', issues.to_string(index=False))
    print('Maintenance history:\n', history.to_string(index=False))
    print('Service history:\n', service.to_string(index=False))
    print('EDA saved to:',out.resolve())

if __name__=='__main__':
    main()
