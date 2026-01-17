# 浙江共同富裕示范区政策效应评估
## 基于合成控制法（Synthetic Control Method）的实证研究

---

## 一、研究背景

2021年6月，中共中央、国务院发布《关于支持浙江高质量发展建设共同富裕示范区的意见》，浙江成为全国唯一的共同富裕示范区。本研究采用合成控制法（SCM）评估该政策对浙江经济社会发展的因果效应。

### 政策时间线
- **2021年6月10日**：中央正式批复浙江建设共同富裕示范区
- **2021年7月19日**：《浙江高质量发展建设共同富裕示范区实施方案（2021—2025年）》发布
- **处理时点**：2021年（本研究设定）

---

## 二、研究设计

### 2.1 合成控制法原理

合成控制法（Abadie et al., 2003, 2010, 2015）通过对未受政策影响的控制单位进行加权组合，构建一个"合成控制组"来模拟处理单位在没有政策干预情况下的反事实结果。

**核心思想**：
$$Y_{1t}^N = \sum_{j=2}^{J+1} w_j Y_{jt}$$

其中，$Y_{1t}^N$ 是浙江在没有政策时的反事实结果，$w_j$ 是各控制省份的权重。

**政策效应估计**：
$$\hat{\tau}_{1t} = Y_{1t} - \hat{Y}_{1t}^N$$

### 2.2 变量设置

| 变量类型 | 变量名称 | 说明 |
|----------|----------|------|
| 结果变量 | 城乡收入比 | 城镇居民可支配收入/农村居民可支配收入 |
| 结果变量 | 人均GDP | 反映"富裕"维度 |
| 结果变量 | 基尼系数 | 反映"共同"维度（如有数据） |
| 预测变量 | 城镇化率 | 常住人口城镇化率 |
| 预测变量 | 第三产业占比 | 产业结构指标 |
| 预测变量 | 人均财政支出 | 政府能力指标 |
| 预测变量 | 人均固定资产投资 | 投资水平 |
| 预测变量 | 人均社会消费品零售额 | 消费水平 |

### 2.3 样本选择

- **处理组**：浙江省
- **控制组候选**：除浙江外的30个省级单位
- **排除标准**：
  - 排除直辖市（北京、天津、上海、重庆）：经济结构特殊
  - 排除西藏、新疆：数据可得性和特殊政策
- **最终控制池**：24个省份
- **时间跨度**：2010-2023年

---

## 三、项目结构

```
scm_zhejiang_prosperity/
├── README.md                 # 项目说明
├── requirements.txt          # Python依赖
├── data/
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后数据
│   └── province_panel.csv    # 省级面板数据
├── src/
│   ├── __init__.py
│   ├── data_collection.py    # 数据收集与处理
│   ├── scm_analysis.py       # 合成控制法核心实现
│   ├── visualization.py      # 可视化模块
│   └── robustness.py         # 稳健性检验
├── results/
│   ├── figures/              # 图表输出
│   └── tables/               # 表格输出
├── docs/
│   └── research_design.md    # 详细研究设计
└── run_analysis.py           # 主运行脚本
```

---

## 四、快速开始

### 4.1 环境配置

```bash
# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 4.2 运行分析

```bash
# 完整分析流程
python run_analysis.py

# 或分步运行
python src/data_collection.py    # 1. 数据准备
python src/scm_analysis.py       # 2. SCM分析
python src/visualization.py      # 3. 可视化
python src/robustness.py         # 4. 稳健性检验
```

---

## 五、数据来源

| 数据 | 来源 | 获取方式 |
|------|------|----------|
| 省级GDP、人均GDP | 国家统计局 | https://data.stats.gov.cn |
| 城乡居民收入 | 国家统计局 | 中国统计年鉴 |
| 城镇化率 | 国家统计局 | 中国统计年鉴 |
| 财政收支 | 财政部 | 中国财政年鉴 |
| 产业结构 | 国家统计局 | 中国统计年鉴 |

**注**：本项目提供模拟数据用于方法演示，实际研究请使用真实数据。

---

## 六、预期输出

### 6.1 主要图表

1. **合成控制对比图**：浙江实际值 vs 合成浙江
2. **政策效应图**：处理效应随时间变化
3. **权重分布图**：各控制省份的权重
4. **安慰剂检验图**：假设其他省份为处理组的分布

### 6.2 主要表格

1. 预测变量平衡表
2. 控制省份权重表
3. 政策效应估计表
4. 稳健性检验结果表

---

## 七、方法拓展

### 7.1 增强型合成控制（Augmented SCM）

```python
# 使用结果回归增强SCM
from src.scm_analysis import AugmentedSCM
model = AugmentedSCM(outcome_model='ridge')
model.fit(data, treated='浙江', time_treat=2021)
```

### 7.2 县级合成控制

```python
# 评估浙江某县试点政策效应
from src.scm_analysis import CountySCM
model = CountySCM()
model.fit(county_data, treated='德清县', time_treat=2019)
```

---

## 八、参考文献

1. Abadie, A., & Gardeazabal, J. (2003). The economic costs of conflict: A case study of the Basque Country. *American Economic Review*, 93(1), 113-132.

2. Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies: Estimating the effect of California's tobacco control program. *Journal of the American Statistical Association*, 105(490), 493-505.

3. Abadie, A., Diamond, A., & Hainmueller, J. (2015). Comparative politics and the synthetic control method. *American Journal of Political Science*, 59(2), 495-510.

4. Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and methodological aspects. *Journal of Economic Literature*, 59(2), 391-425.

---

## 九、作者与许可

- **研究目的**：学术研究与方法演示
- **许可协议**：MIT License
- **联系方式**：[您的邮箱]

---

## 十、更新日志

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2026-01-17 | v1.0 | 初始版本，包含基础SCM实现 |
