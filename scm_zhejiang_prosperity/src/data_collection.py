"""
数据收集与处理模块
=====================

本模块提供：
1. 省级面板分析的元数据常量（处理省份、控制省份、变量定义）
2. 真实数据获取指南
3. 真实数据的预处理与统计工具

本模块不生成任何模拟/合成/随机数据。
分析所需的省级面板数据必须由用户基于《中国统计年鉴》等公开权威来源
准备为 data/province_panel_real.csv，其字段说明见 README.md。
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# =============================================================================
# 一、省份和变量定义
# =============================================================================

# 中国省级单位
PROVINCES = [
    '北京', '天津', '河北', '山西', '内蒙古',
    '辽宁', '吉林', '黑龙江', '上海', '江苏',
    '浙江', '安徽', '福建', '江西', '山东',
    '河南', '湖北', '湖南', '广东', '广西',
    '海南', '重庆', '四川', '贵州', '云南',
    '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆'
]

# 控制组省份（排除直辖市和特殊地区）
CONTROL_PROVINCES = [
    '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江',
    '江苏', '安徽', '福建', '江西', '山东', '河南',
    '湖北', '湖南', '广东', '广西', '海南', '四川',
    '贵州', '云南', '陕西', '甘肃', '青海', '宁夏'
]

# 被排除的省份及原因
EXCLUDED_PROVINCES = {
    '北京': '直辖市，经济结构特殊',
    '天津': '直辖市，经济结构特殊',
    '上海': '直辖市，经济结构特殊',
    '重庆': '直辖市，经济结构特殊',
    '西藏': '数据可得性差，特殊政策',
    '新疆': '特殊政策环境'
}

# 处理省份
TREATED_PROVINCE = '浙江'

# 处理年份
TREATMENT_YEAR = 2021

# 分析变量
OUTCOME_VARIABLES = {
    'urban_rural_income_ratio': '城乡收入比',
    'gdp_per_capita': '人均GDP（万元）',
    'rural_income': '农村居民人均可支配收入（万元）',
    'urban_income': '城镇居民人均可支配收入（万元）'
}

PREDICTOR_VARIABLES = {
    'urbanization_rate': '城镇化率（%）',
    'tertiary_share': '第三产业占比（%）',
    'fiscal_expenditure_pc': '人均财政支出（万元）',
    'fixed_investment_pc': '人均固定资产投资（万元）',
    'retail_sales_pc': '人均社会消费品零售额（万元）',
    'education_years': '平均受教育年限',
    'population_density': '人口密度（人/平方公里）'
}


# =============================================================================
# 二、（已删除）模拟数据生成
# =============================================================================
#
# 本项目此前在该位置提供 generate_simulated_data / save_simulated_data 用于
# 方法演示。出于学术诚信考虑，已彻底删除所有模拟与随机数据生成逻辑——
# 本模块只接受用户提供的真实数据。
# 如需准备数据，请参阅 README.md 与 print_data_guide() 中的说明。


# =============================================================================
# 三、真实数据获取指南
# =============================================================================

REAL_DATA_GUIDE = """
================================================================================
                        真实数据获取指南
================================================================================

【数据来源1】国家统计局数据库
--------------------------
网址: https://data.stats.gov.cn/
路径: 年度数据 → 分省年度数据

可获取指标:
- 地区生产总值
- 人均地区生产总值  
- 城镇居民人均可支配收入
- 农村居民人均可支配收入
- 年末常住人口
- 城镇化率

【数据来源2】中国统计年鉴
--------------------------
网址: http://www.stats.gov.cn/sj/ndsj/
格式: PDF/Excel

可获取指标:
- 各省主要经济指标
- 城乡居民收入
- 财政收支
- 固定资产投资

【数据来源3】EPS数据平台
--------------------------
网址: http://www.epsnet.com.cn/
数据库: 中国区域经济数据库

优势: 
- 数据整合度高
- 时间序列完整
- 支持批量下载

【数据来源4】CEIC数据库
--------------------------
适合高校用户，数据质量高

【数据处理注意事项】
--------------------------
1. 价格指标需要用CPI/GDP平减指数调整为实际值
2. 2020年数据可能受疫情影响，需注意处理
3. 部分省份行政区划调整需要统一口径
4. 缺失值处理：线性插值或均值填补

【变量构建公式】
--------------------------
城乡收入比 = 城镇居民人均可支配收入 / 农村居民人均可支配收入
人均GDP = GDP / 年末常住人口
城镇化率 = 城镇常住人口 / 常住总人口 × 100%

================================================================================
"""


def print_data_guide():
    """打印真实数据获取指南"""
    print(REAL_DATA_GUIDE)


# =============================================================================
# 四、数据预处理函数
# =============================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据预处理
    
    处理内容：
    1. 缺失值处理
    2. 异常值检测
    3. 变量标准化（可选）
    """
    df = df.copy()
    
    # 检查缺失值
    missing_summary = df.isnull().sum()
    if missing_summary.any():
        print("缺失值统计:")
        print(missing_summary[missing_summary > 0])
        
        # 对数值列进行线性插值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df.groupby('province')[col].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
    
    # 异常值检测（IQR方法）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['treated', 'post', 'year']:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            warnings.warn(f"变量 {col} 存在 {len(outliers)} 个潜在异常值")
    
    return df


def reshape_for_scm(
    df: pd.DataFrame,
    outcome_var: str,
    unit_col: str = 'province',
    time_col: str = 'year'
) -> pd.DataFrame:
    """
    将长面板数据转换为SCM所需的宽格式
    
    参数:
    -----
    df: 长格式面板数据
    outcome_var: 结果变量名
    unit_col: 单位列名
    time_col: 时间列名
    
    返回:
    -----
    DataFrame: 宽格式数据（行=时间，列=单位）
    """
    wide_df = df.pivot(
        index=time_col,
        columns=unit_col,
        values=outcome_var
    )
    return wide_df


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """生成描述性统计"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['year', 'treated', 'post']]
    
    summary = df[numeric_cols].describe().T
    summary['missing'] = df[numeric_cols].isnull().sum()
    summary['missing_pct'] = summary['missing'] / len(df) * 100
    
    return summary.round(3)


# =============================================================================
# 五、主函数
# =============================================================================

def main():
    """数据准备主函数：仅打印数据获取指南。

    本项目不再生成任何模拟数据；分析所需的省级面板数据必须由研究者基于
    《中国统计年鉴》《浙江统计年鉴》等公开权威来源整理为
    data/province_panel_real.csv。具体字段见 README.md。
    """
    print("=" * 60)
    print("浙江共同富裕示范区政策效应评估 - 数据准备")
    print("=" * 60)

    print_data_guide()

    print("\n请基于上述真实数据来源整理 data/province_panel_real.csv，")
    print("字段说明见 README.md《数据准备》一节。本项目不提供任何模拟数据。")


if __name__ == "__main__":
    main()
