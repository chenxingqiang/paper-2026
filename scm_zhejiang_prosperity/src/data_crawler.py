"""
省级面板数据爬虫
=================

从国家统计局（NBS）`data.stats.gov.cn` 的「分省年度数据」库自动抓取
合成控制法分析所需的全部字段，落盘为：

* `data/raw/<indicator>.csv`  —— 每个原始指标一份长表（保留 NBS 原值与单位）
* `data/province_panel_real.csv`  —— 拼装好的最终面板（符合 README schema）

使用方法
--------

```
# 在国内可访问 data.stats.gov.cn 的环境中执行
pip install akshare>=1.18
python -m src.data_crawler --start 2010 --end 2024 \
    --out data/province_panel_real.csv
```

设计要点
--------

1. 仅使用 NBS 公开 API（通过 `akshare.macro_china_nbs_region` 封装），
   所有原值可追溯到 NBS 同一栏目。**不生成任何模拟/外推数值**。
2. NBS 服务器侧 WAF 对境外 IP 段会返回 403/UrlACL，因此必须在国内
   网络环境运行；若检测到网络失败，脚本会清晰报错并指引。
3. 字段单位严格遵循 `data/README.md`：
   * 收入、人均 GDP、人均财政支出、人均固投、人均社零均为「万元」
     （NBS 原值为元/亿元，本脚本统一换算）
   * `urbanization_rate`、`tertiary_share` 为百分比 (%)
4. `urban_rural_income_ratio = urban_income / rural_income`
   `treated = (province == "浙江")`，`post = (year >= 2021)`
5. 缺失的 (省份, 年份) 单元 **不** 填充随机数；脚本只会保留缺失，
   并提示用户在 README 的「缺失值处理」流程中自行补全。

NBS 抓取路径与指标映射
----------------------

`akshare.macro_china_nbs_region(kind, path, indicator, region, period)`
对应 NBS 网站「数据查询 → 分省年度数据」中的栏目层级。
本脚本使用的映射在 `INDICATOR_MAP` 中维护，若 NBS 改版或更名，请
修改对应条目（保持单位换算逻辑同步）。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


# 分析目标省份（处理组 + 24 个控制省）
TREATED_PROVINCE = "浙江"
CONTROL_PROVINCES = [
    "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
    "江苏", "安徽", "福建", "江西", "山东", "河南",
    "湖北", "湖南", "广东", "广西", "海南", "四川",
    "贵州", "云南", "陕西", "甘肃", "青海", "宁夏",
]
TARGET_PROVINCES = [TREATED_PROVINCE] + CONTROL_PROVINCES

# NBS region 名称：上述简称需补足为「XX省/自治区」以匹配 NBS 标签
REGION_FULLNAME = {
    "北京": "北京市", "天津": "天津市", "河北": "河北省", "山西": "山西省",
    "内蒙古": "内蒙古自治区", "辽宁": "辽宁省", "吉林": "吉林省",
    "黑龙江": "黑龙江省", "上海": "上海市", "江苏": "江苏省",
    "浙江": "浙江省", "安徽": "安徽省", "福建": "福建省",
    "江西": "江西省", "山东": "山东省", "河南": "河南省",
    "湖北": "湖北省", "湖南": "湖南省", "广东": "广东省",
    "广西": "广西壮族自治区", "海南": "海南省", "重庆": "重庆市",
    "四川": "四川省", "贵州": "贵州省", "云南": "云南省",
    "西藏": "西藏自治区", "陕西": "陕西省", "甘肃": "甘肃省",
    "青海": "青海省", "宁夏": "宁夏回族自治区", "新疆": "新疆维吾尔自治区",
}


@dataclass(frozen=True)
class IndicatorSpec:
    """NBS 指标抓取规格。

    Attributes
    ----------
    field : 目标 schema 字段名（见 data/README.md）
    path  : NBS「分省年度数据」一级栏目（如 "综合"、"人民生活"）
    indicator : NBS 指标的中文全名（含括号里的单位）
    to_target_unit : 单位换算函数：把 NBS 原值换算成 schema 目标单位
    """

    field: str
    path: str
    indicator: str
    to_target_unit: Callable[[float], float]


def _yuan_to_wanyuan(x: float) -> float:
    """元 → 万元"""
    return x / 10_000.0


def _identity(x: float) -> float:
    return x


# 单位换算策略：脚本不直接把「亿元 / 总人口」预先固定为某种公式，
# 而是先抓取原始指标，然后在拼装阶段统一做人均化。这里 IndicatorSpec
# 描述的是「直接从 NBS 取一列值即可使用」的指标。
INDICATOR_MAP: list[IndicatorSpec] = [
    # ── 直接为「元」的人均收入 → 换算成「万元」 ───────────────────────────────
    IndicatorSpec(
        field="urban_income",
        path="人民生活",
        indicator="城镇居民人均可支配收入(元)",
        to_target_unit=_yuan_to_wanyuan,
    ),
    IndicatorSpec(
        field="rural_income",
        path="人民生活",
        indicator="农村居民人均可支配收入(元)",
        to_target_unit=_yuan_to_wanyuan,
    ),
    IndicatorSpec(
        field="gdp_per_capita",
        path="综合",
        indicator="人均地区生产总值(元)",
        to_target_unit=_yuan_to_wanyuan,
    ),
    # ── 直接为「%」的比率 ─────────────────────────────────────────────────────
    IndicatorSpec(
        field="urbanization_rate",
        path="人口",
        indicator="城镇化率(%)",
        to_target_unit=_identity,
    ),
]

# 需要由「亿元总量 / 年末常住人口（万人）」做人均化的指标。
# 公式：人均(万元) = 总量(亿元) * 1e8 / 人口(万人) / 1e4 / 1e4
#                = 总量(亿元) / 人口(万人) (单位：万元/人)
PER_CAPITA_INDICATORS: list[IndicatorSpec] = [
    IndicatorSpec(
        field="fiscal_expenditure_pc",
        path="财政",
        indicator="一般公共预算支出(亿元)",
        to_target_unit=_identity,  # 在 _build_panel 中再做人均化
    ),
    IndicatorSpec(
        field="fixed_investment_pc",
        path="固定资产投资和房地产",
        indicator="全社会固定资产投资(亿元)",
        to_target_unit=_identity,
    ),
    IndicatorSpec(
        field="retail_sales_pc",
        path="国内贸易",
        indicator="社会消费品零售总额(亿元)",
        to_target_unit=_identity,
    ),
]

# 第三产业占比 = 第三产业增加值(亿元) / 地区生产总值(亿元) * 100
# 因此还需要这两个分子分母指标。
SHARE_NUMERATOR = IndicatorSpec(
    field="_tertiary_value",
    path="国民经济核算",
    indicator="第三产业增加值(亿元)",
    to_target_unit=_identity,
)
SHARE_DENOMINATOR = IndicatorSpec(
    field="_gdp_total",
    path="综合",
    indicator="地区生产总值(亿元)",
    to_target_unit=_identity,
)

# 年末常住人口（万人），用于把亿元总量换算成人均
POPULATION = IndicatorSpec(
    field="_population_wan",
    path="人口",
    indicator="年末常住人口(万人)",
    to_target_unit=_identity,
)


# =============================================================================
# 抓取
# =============================================================================

def _fetch_one(spec: IndicatorSpec, start: int, end: int, retry: int = 3,
               sleep_sec: float = 1.0) -> pd.DataFrame:
    """抓取单个指标的全部目标省份数据。

    返回 long 格式：columns=['province', 'year', '<spec.field>']，单位为
    NBS 原值（不做换算；换算延后到 _build_panel 中统一处理，便于审计）。
    """
    try:
        import akshare as ak  # 延迟 import
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖 akshare。请先：pip install akshare>=1.18"
        ) from exc

    period = f"{start}-{end}"
    rows: list[dict] = []
    for short in TARGET_PROVINCES:
        full = REGION_FULLNAME[short]
        last_err: Optional[Exception] = None
        for attempt in range(1, retry + 1):
            try:
                df = ak.macro_china_nbs_region(
                    kind="分省年度数据",
                    path=spec.path,
                    indicator=spec.indicator,
                    region=full,
                    period=period,
                )
                break
            except Exception as exc:  # NBS WAF / 网络故障 / 解析失败
                last_err = exc
                if attempt < retry:
                    time.sleep(sleep_sec * attempt)
                else:
                    raise RuntimeError(
                        f"抓取 [{spec.path}/{spec.indicator}] 省份={full} "
                        f"失败：{type(exc).__name__}: {exc}\n"
                        f"提示：data.stats.gov.cn 对境外 IP 会返回 403/UrlACL，"
                        f"请在国内网络环境运行本脚本。"
                    ) from last_err

        if df is None or df.empty:
            print(f"  ! 空响应：{spec.indicator} - {full}", file=sys.stderr)
            continue

        # akshare 返回列名一般为 ['时间', '<indicator>'] 或 ['时间', '指标', '数值']
        df = df.copy()
        time_col = next(
            (c for c in df.columns if "时间" in c or c == "year"),
            None,
        )
        if time_col is None:
            raise RuntimeError(
                f"无法识别时间列，akshare 返回 columns={list(df.columns)}；"
                f"NBS 接口可能已改版，请更新 data_crawler.py。"
            )
        # 找到值列：除时间列外的第一个数值列
        value_col = None
        for c in df.columns:
            if c == time_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                value_col = c
                break
        if value_col is None:
            # 兜底：最后一列尝试转 float
            value_col = df.columns[-1]
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

        df["year"] = (
            df[time_col].astype(str).str.extract(r"(\d{4})")[0].astype("Int64")
        )
        df["province"] = short
        df = df.dropna(subset=["year"]).rename(columns={value_col: spec.field})
        rows.append(df[["province", "year", spec.field]])

        time.sleep(sleep_sec)  # 礼貌限速

    if not rows:
        raise RuntimeError(f"指标 {spec.indicator} 全部省份均无数据。")
    out = pd.concat(rows, ignore_index=True)
    out["year"] = out["year"].astype(int)
    out[spec.field] = pd.to_numeric(out[spec.field], errors="coerce")
    return out


def _save_raw(df: pd.DataFrame, name: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → 保存原始指标 {name}: {path}  (rows={len(df)})")


# =============================================================================
# 拼装最终面板
# =============================================================================

def _build_panel(start: int, end: int) -> pd.DataFrame:
    """串联所有指标，构建最终 schema。"""
    print("\n[1/3] 抓取直接指标 ...")
    direct: list[pd.DataFrame] = []
    for spec in INDICATOR_MAP:
        print(f"  - {spec.field} ← {spec.path}/{spec.indicator}")
        df = _fetch_one(spec, start, end)
        _save_raw(df, spec.field)
        df[spec.field] = df[spec.field].astype(float).map(spec.to_target_unit)
        direct.append(df)

    print("\n[2/3] 抓取人均化与占比所需的总量/人口指标 ...")
    pop_df = _fetch_one(POPULATION, start, end)
    _save_raw(pop_df, POPULATION.field)
    gdp_df = _fetch_one(SHARE_DENOMINATOR, start, end)
    _save_raw(gdp_df, SHARE_DENOMINATOR.field)
    ter_df = _fetch_one(SHARE_NUMERATOR, start, end)
    _save_raw(ter_df, SHARE_NUMERATOR.field)

    pc_dfs: list[pd.DataFrame] = []
    for spec in PER_CAPITA_INDICATORS:
        print(f"  - {spec.field} ← {spec.path}/{spec.indicator} (亿元 → 万元/人)")
        total = _fetch_one(spec, start, end)
        _save_raw(total, f"{spec.field}__total_yiyuan")
        merged = total.merge(pop_df, on=["province", "year"], how="left")
        # 亿元 / 万人 = 万元 / 人  （1e8 / 1e4 / 1e4 = 1）
        merged[spec.field] = merged[spec.field] / merged[POPULATION.field]
        pc_dfs.append(merged[["province", "year", spec.field]])

    print("\n[3/3] 合并面板并构造派生字段 ...")
    panel = direct[0]
    for df in direct[1:] + pc_dfs:
        panel = panel.merge(df, on=["province", "year"], how="outer")

    # 第三产业占比
    share = ter_df.merge(gdp_df, on=["province", "year"], how="outer")
    share["tertiary_share"] = (
        share[SHARE_NUMERATOR.field] / share[SHARE_DENOMINATOR.field] * 100.0
    )
    panel = panel.merge(
        share[["province", "year", "tertiary_share"]],
        on=["province", "year"],
        how="outer",
    )

    # 派生字段
    panel["urban_rural_income_ratio"] = (
        panel["urban_income"] / panel["rural_income"]
    )
    panel["treated"] = (panel["province"] == TREATED_PROVINCE).astype(int)
    panel["post"] = (panel["year"] >= 2021).astype(int)

    column_order = [
        "province", "year",
        "urban_rural_income_ratio",
        "gdp_per_capita",
        "urban_income", "rural_income",
        "urbanization_rate", "tertiary_share",
        "fiscal_expenditure_pc", "fixed_investment_pc", "retail_sales_pc",
        "treated", "post",
    ]
    panel = panel[column_order].sort_values(["province", "year"]).reset_index(drop=True)
    return panel


# =============================================================================
# CLI
# =============================================================================

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="从国家统计局抓取省级面板数据并保存为 province_panel_real.csv",
    )
    parser.add_argument("--start", type=int, default=2010, help="起始年份（含）")
    parser.add_argument("--end", type=int, default=2024, help="终止年份（含）")
    parser.add_argument(
        "--out",
        type=str,
        default=str(DATA_DIR / "province_panel_real.csv"),
        help="输出 CSV 路径",
    )
    args = parser.parse_args(argv)

    print("=" * 64)
    print(f" NBS 省级面板抓取  {args.start}–{args.end}")
    print(f" 处理省份: {TREATED_PROVINCE}（处理组）+ {len(CONTROL_PROVINCES)} 控制省")
    print("=" * 64)
    print(
        "提示：data.stats.gov.cn 对境外 IP 段会返回 403/UrlACL，"
        "本脚本必须在国内网络环境运行。\n"
    )

    panel = _build_panel(args.start, args.end)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False, encoding="utf-8-sig")

    n_missing = panel.drop(columns=["treated", "post"]).isna().any(axis=1).sum()
    print("\n" + "=" * 64)
    print(f" 完成：{out_path}")
    print(f"  rows={len(panel)}  provinces={panel['province'].nunique()}"
          f"  years={panel['year'].min()}–{panel['year'].max()}")
    print(f"  含缺失的行数: {n_missing}（请按 data/README.md 第 3 节处理）")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
