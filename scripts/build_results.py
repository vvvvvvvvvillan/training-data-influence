"""Validate final results and regenerate all tables, plots, and README. No training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
METHODS=['random','loss','if_original','tracin_self']
LABELS=['Random','Loss','Original IF','TracIn']
CONDITIONS=[('symmetric',.2),('symmetric',.4),('pairflip',.2),('pairflip',.4),('instance',.2),('instance',.4)]
NAMES={'symmetric':'随机错标','pairflip':'类别依赖错标','instance':'实例依赖错标'}
df=pd.read_csv(ROOT/'data/results.csv')
keys=['noise','rate','seed','method','action']
expected={(n,r,s,m,a) for n,r in CONDITIONS for s in range(200,205) for m in METHODS for a in ['correct','remove']}
assert len(df)==240 and set(df[keys].itertuples(index=False,name=None))==expected
assert not df[keys].duplicated().any()
assert np.isfinite(df.select_dtypes(include='number').to_numpy()).all()
assert np.allclose(df.test_accuracy-df.base_test_accuracy,df.accuracy_delta)
assert (df.budget==1000).all()
for metric in ['AP','AUROC','precision_at_budget','recall_at_budget','test_accuracy','base_test_accuracy']:
 assert df[metric].between(0,1).all()
assert np.allclose(df.hits/1000,df.precision_at_budget)
assert np.allclose(df.hits/(df.rate*10000),df.recall_at_budget)
for _,g in df.groupby(['noise','rate','seed','method']):
 assert g.AP.nunique()==1
metrics=['AP','AUROC','precision_at_budget','recall_at_budget','test_accuracy','test_loss','accuracy_delta']
summary=df.groupby(['noise','rate','method','action'])[metrics].agg(['mean','std'])
summary.to_csv(ROOT/'data/summary.csv')
(ROOT/'data/validation.json').write_text(json.dumps({'passed':True,'cases':30,'methods':4,'cleaning_results':240,'checks':['complete unique matrix','finite metrics','accuracy delta','budget and hit consistency','detection consistency']},indent=2)+'\n')

def table(metric,action,scale=1):
 lines=['| 噪声 | 比例 | Random | Loss | 原始 IF | TracIn |','|---|---:|---:|---:|---:|---:|']
 for n,r in CONDITIONS:
  vals=[]
  for m in METHODS:
   g=df[(df.noise==n)&(df.rate==r)&(df.method==m)&(df.action==action)][metric]*scale
   vals.append(f'{g.mean():.2f} ± {g.std(ddof=1):.2f}' if scale==100 else f'{g.mean():.4f} ± {g.std(ddof=1):.4f}')
  lines.append(f'| {NAMES[n]} | {r:.0%} | '+' | '.join(vals)+' |')
 return '\n'.join(lines)

for metric,action,file,ylabel,scale in [('AP','correct','detection_ap','Average precision',1),('accuracy_delta','correct','correction_gain','Accuracy gain (percentage points)',100),('accuracy_delta','remove','removal_gain','Accuracy gain (percentage points)',100)]:
 fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
 for ax,(n,r) in zip(axes.flat,CONDITIONS):
  groups=[df[(df.noise==n)&(df.rate==r)&(df.method==m)&(df.action==action)][metric]*scale for m in METHODS]
  ax.bar(LABELS,[g.mean() for g in groups],yerr=[g.std(ddof=1) for g in groups],capsize=3)
  ax.axhline(r if metric=='AP' else 0,color='gray',linestyle='--');ax.set_title(f'{n} {r:.0%}');ax.set_ylabel(ylabel)
  if metric=='AP':ax.set_ylim(0,1)
 fig.savefig(ROOT/f'figures/{file}.png',dpi=180);plt.close(fig)
readme='''# Training Data Influence for Detecting Mislabeled Samples

CIFAR-10 / ResNet-18 课程项目最终结果仓库。比较 **Random、Training Loss、原始最后一层 Influence Functions、TracInCP 最后一层自影响** 四种方法。

## 主要结论

- TracIn 在六个设置中的五个取得这四种方法里最高的平均检测 AP；40% 类别依赖错标下原始 IF 最高。
- 在固定检查1,000张并恢复真实标签后，TracIn同样在五个设置取得最高平均测试准确率，40%类别依赖错标下原始IF最好。
- 方法优势依赖噪声结构；检测 AP 高不保证清理后准确率一定更高。报告均值和跨种子标准差，不将小幅均值差自动解释为统计显著。
- IF使用额外干净验证标签，TracIn使用训练历史，信息预算不同；结果不能仅归因于算法公式。

## 实验设计

| 项目 | 设置 |
|---|---|
| 数据集 | CIFAR-10，训练子集10,000张，官方测试集10,000张 |
| 网络 | CIFAR适配ResNet-18，从头训练；3×3 stem，无初始maxpool |
| 噪声 | 随机、类别依赖、实例依赖，各20%/40% |
| 随机种子 | 200–204，共5个 |
| 基础设置 | 3×2×5=30组 |
| 审核预算 | 每种方法选取前1,000张，训练集的10% |
| 清理动作 | oracle标签修正、删除；各从相同seed初始化完整重训 |
| 最终结果 | 4方法×2动作×30组=240行 |
| 训练 | SGD，batch128，7,900步，lr0.1，warmup5轮后cosine，FP32 |
| 干净参考 | 原实验5个干净模型的平均测试准确率86.63% |

详细方法、数据隔离及复现边界见[实验协议](docs/PROTOCOL.md)。

## 检测 AP

五种子均值 ± 样本标准差，越高越好。Random的参考水平随错误标签比例变化。

'''+table('AP','correct')+'''

![检测AP](figures/detection_ap.png)

## 修正后测试准确率（%）

修正使用真实标签模拟人工审核，不是自动预测新标签。

'''+table('test_accuracy','correct',100)+'''

## 修正后相对基础模型的提升（百分点）

'''+table('accuracy_delta','correct',100)+'''

![修正提升](figures/correction_gain.png)

## 删除后测试准确率（%）

'''+table('test_accuracy','remove',100)+'''

![删除提升](figures/removal_gain.png)

图中误差线均为跨种子SD，不是置信区间。TracIn在20%随机错标下删除准确率78.27%、修正78.21%，差异很小，因此不能声称修正始终优于删除。

## 成本与实用性

| 方法 | 主要额外成本 | 本次30组实际记录 |
|---|---|---|
| Random | 随机排序 | 未独立计时，计算很少 |
| Loss | 最终模型的一次前向评价 | 未独立计时 |
| 原始IF | 特征、干净验证梯度、头部CG | 原实验共享特征33.47秒；共享评分批次11.32秒，不能当作该方法独立耗时 |
| TracIn | 9个历史checkpoint的特征和梯度评分 | 为补齐历史轨迹补训3.13小时；评分未独立计时 |

IF需要额外2,000个干净归因标签。TracIn若在原训练中预先保存checkpoint，可避免本次轨迹补训，但仍有评分与存储成本。每种方法60次清理重训约6–7小时，这主要是比较方法效果的实验成本。实际硬件为单个H100 MIG 3g.40gb，不是8卡运行。

## 文件与复现

- [data/results.csv](data/results.csv)：240条完整逐种子结果，含AP、AUROC、P/R@10%、accuracy、test loss和耗时。
- [data/summary.csv](data/summary.csv)：各指标均值和样本标准差。
- [data/validation.json](data/validation.json)：结果矩阵与指标一致性验证。
- [config/training.json](config/training.json)：原冻结训练参数。
- [config/splits_manifest.json](config/splits_manifest.json)：数据及划分身份哈希。
- [data/provenance.json](data/provenance.json)：源表SHA256和保留方法。
- [scripts/build_results.py](scripts/build_results.py)：重建此README、统计表和图，并验证数据。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_results.py
```

本仓库定位为**最终结果与统计复现仓库**：上述命令重算表格和图，不重新训练模型。原始训练源码、图像数据、逐样本分数及模型权重仍保留在原项目/服务器，未打包进本仓库。不能仅凭汇总结果重新核验每张图的AP排序。

## 局限

仅一个数据集、固定划分和预算、五个种子；噪声为合成噪声。两种影响方法都仅使用最后分类层，不能代表全网络影响估计。最终四方法汇总是在完整实验之后整理形成的描述性比较，不是新的预注册显著性检验。未对多重比较作校正；不宣称某方法在所有任务中最好。
'''
(ROOT/'README.md').write_text(readme)
print('Validated 240 rows; generated README, summary, and three figures.')
