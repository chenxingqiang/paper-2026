# 数据准备说明

本项目**不**生成任何模拟、合成或随机数据。运行 `run_analysis.py` 之前，
必须由研究者基于公开权威来源整理真实的省级面板数据，并保存为
`data/province_panel_real.csv`。

## 1. 数据来源（必须真实）

| 来源 | 网址 | 说明 |
|------|------|------|
| 国家统计局数据库 | <https://data.stats.gov.cn/> | 分省年度数据 |
| 中国统计年鉴 | <http://www.stats.gov.cn/sj/ndsj/> | 各年版 PDF / Excel |
| 浙江统计年鉴 | <http://tjj.zj.gov.cn/> | 浙江省统计局 |
| EPS 数据平台 | <http://www.epsnet.com.cn/> | 中国区域经济数据库 |
| CEIC 数据库 | — | 高校用户可订阅 |

任何用于合成控制法估计的数值都必须可追溯到上述（或同等权威的）公开统计来源；
**禁止使用任何形式的模拟、外推或加噪数值**填充缺失年份/省份。

## 2. CSV 字段规范

`data/province_panel_real.csv` 需采用 UTF-8 编码（推荐 `utf-8-sig`），
字段如下：

| 字段 | 类型 | 含义 | 单位 |
|------|------|------|------|
| `province` | str | 省份名（简称，例如 `浙江`、`江苏`） | — |
| `year` | int | 年份 | — |
| `urban_rural_income_ratio` | float | 城乡收入比 = 城镇居民人均可支配收入 / 农村居民人均可支配收入 | 倍数 |
| `gdp_per_capita` | float | 人均地区生产总值 | 万元 |
| `urban_income` | float | 城镇居民人均可支配收入 | 万元 |
| `rural_income` | float | 农村居民人均可支配收入 | 万元 |
| `urbanization_rate` | float | 常住人口城镇化率 | % |
| `tertiary_share` | float | 第三产业增加值占 GDP 比重 | % |
| `fiscal_expenditure_pc` | float | 人均一般公共预算支出 | 万元 |
| `fixed_investment_pc` | float | 人均固定资产投资 | 万元 |
| `retail_sales_pc` | float | 人均社会消费品零售总额 | 万元 |
| `treated` | int | 处理组指示，浙江=1，其他=0 | — |
| `post` | int | 处理后指示，year ≥ 2021 = 1，其他=0 | — |

至少需覆盖：

* **处理组**：浙江
* **控制组候选**：河北、山西、内蒙古、辽宁、吉林、黑龙江、江苏、安徽、福建、
  江西、山东、河南、湖北、湖南、广东、广西、海南、四川、贵州、云南、
  陕西、甘肃、青海、宁夏（24 个；排除直辖市与西藏/新疆）
* **时间跨度**：2010 年至最新可得年份（建议至少到 2023 年）

## 3. 缺失值处理

如个别 (省份, 年份) 单元因官方口径调整而暂缺，请优先：

1. 查阅当年《中国统计年鉴》《地方统计年鉴》补全；
2. 仍缺失时，使用 `src/data_collection.py::preprocess_data` 中的线性插值
   作为退而求其次的处理，并在论文中明确说明。

**严禁**用任何随机噪声、基线外推或人为构造的数值填补。

## 4. 数据验证

CSV 准备完成后，运行：

```bash
python run_analysis.py --data data/province_panel_real.csv --quick
```

脚本会输出样本量、省份数和时间跨度，请人工核对与官方统计数据是否一致。

## 5. 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `data/province_panel_real.csv` | 用户提供 | 真实省级面板数据，必填 |
| `data/raw/` | 可选 | 用于存放从国家统计局 / akshare 等下载的原始 csv/xlsx |

历史版本中曾存在的 `province_panel.csv`（纯模拟）和包含 `np.random` 外推的
`province_panel_real.csv` 已被彻底删除。
