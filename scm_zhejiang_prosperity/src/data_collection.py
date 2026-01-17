"""
数据收集与处理模块
=====================

本模块提供：
1. 真实数据获取指南
2. 模拟数据生成（用于方法演示）
3. 数据预处理函数
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
# 二、模拟数据生成
# =============================================================================

def generate_simulated_data(
    start_year: int = 2010,
    end_year: int = 2023,
    treatment_year: int = 2021,
    treatment_effect: float = -0.05,  # 城乡收入比下降5%
    seed: int = 42
) -> pd.DataFrame:
    """
    生成模拟的省级面板数据
    
    参数:
    -----
    start_year: 起始年份
    end_year: 结束年份
    treatment_year: 政策实施年份
    treatment_effect: 政策效应（对城乡收入比的影响）
    seed: 随机种子
    
    返回:
    -----
    DataFrame: 省级面板数据
    """
    np.random.seed(seed)
    
    years = list(range(start_year, end_year + 1))
    n_years = len(years)
    
    # 所有省份（包括处理组和控制组）
    all_provinces = [TREATED_PROVINCE] + CONTROL_PROVINCES
    n_provinces = len(all_provinces)
    
    # 省份固定效应（基于实际经济发展水平的大致排序）
    province_effects = {
        '浙江': 0.95,      # 发达省份
        '江苏': 0.98,
        '广东': 0.92,
        '山东': 0.85,
        '福建': 0.82,
        '辽宁': 0.70,
        '湖北': 0.68,
        '湖南': 0.65,
        '河北': 0.62,
        '安徽': 0.60,
        '江西': 0.58,
        '河南': 0.56,
        '四川': 0.55,
        '陕西': 0.52,
        '山西': 0.50,
        '内蒙古': 0.48,
        '吉林': 0.45,
        '黑龙江': 0.43,
        '广西': 0.42,
        '云南': 0.40,
        '贵州': 0.38,
        '甘肃': 0.35,
        '海南': 0.55,
        '青海': 0.32,
        '宁夏': 0.38
    }
    
    data_list = []
    
    for province in all_provinces:
        # 省份基础效应
        base_effect = province_effects.get(province, 0.5)
        
        for i, year in enumerate(years):
            # 时间趋势
            time_trend = i / n_years
            
            # 是否为处理期
            post_treatment = (province == TREATED_PROVINCE) and (year >= treatment_year)
            
            # 1. 城乡收入比（主要结果变量）
            # 基础值约2.5-3.0，随时间下降
            base_ratio = 3.2 - 0.8 * base_effect - 0.3 * time_trend
            noise = np.random.normal(0, 0.05)
            
            # 浙江政策效应（逐年增强）
            if post_treatment:
                years_since_treatment = year - treatment_year + 1
                policy_effect = treatment_effect * min(years_since_treatment, 3) / 3
            else:
                policy_effect = 0
            
            urban_rural_ratio = base_ratio + noise + policy_effect
            
            # 2. 人均GDP（万元）
            base_gdp = 3 + 8 * base_effect + 4 * time_trend + 0.5 * time_trend ** 2
            gdp_per_capita = base_gdp * (1 + np.random.normal(0, 0.03))
            if post_treatment:
                gdp_per_capita *= (1 + 0.02 * (year - treatment_year + 1))  # 政策促进增长
            
            # 3. 城镇化率（%）
            base_urban = 45 + 20 * base_effect + 15 * time_trend
            urbanization_rate = min(base_urban + np.random.normal(0, 1), 85)
            
            # 4. 第三产业占比（%）
            base_tertiary = 40 + 15 * base_effect + 10 * time_trend
            tertiary_share = min(base_tertiary + np.random.normal(0, 1.5), 70)
            
            # 5. 人均财政支出（万元）
            base_fiscal = 0.5 + 1.5 * base_effect + 0.8 * time_trend
            fiscal_expenditure_pc = base_fiscal * (1 + np.random.normal(0, 0.05))
            
            # 6. 人均固定资产投资（万元）
            base_investment = 2 + 5 * base_effect + 3 * time_trend
            fixed_investment_pc = base_investment * (1 + np.random.normal(0, 0.08))
            
            # 7. 人均社会消费品零售额（万元）
            base_retail = 1 + 3 * base_effect + 2 * time_trend
            retail_sales_pc = base_retail * (1 + np.random.normal(0, 0.04))
            
            # 8. 城镇居民人均可支配收入（万元）
            base_urban_income = 2 + 4 * base_effect + 2.5 * time_trend
            urban_income = base_urban_income * (1 + np.random.normal(0, 0.03))
            
            # 9. 农村居民人均可支配收入（万元）
            rural_income = urban_income / urban_rural_ratio
            
            # 10. 平均受教育年限
            base_edu = 8 + 2.5 * base_effect + 0.8 * time_trend
            education_years = base_edu + np.random.normal(0, 0.2)
            
            # 11. 人口密度
            base_density = 200 + 400 * base_effect
            population_density = base_density * (1 + np.random.normal(0, 0.02))
            
            data_list.append({
                'province': province,
                'year': year,
                'urban_rural_income_ratio': round(urban_rural_ratio, 3),
                'gdp_per_capita': round(gdp_per_capita, 2),
                'urban_income': round(urban_income, 2),
                'rural_income': round(rural_income, 2),
                'urbanization_rate': round(urbanization_rate, 1),
                'tertiary_share': round(tertiary_share, 1),
                'fiscal_expenditure_pc': round(fiscal_expenditure_pc, 3),
                'fixed_investment_pc': round(fixed_investment_pc, 2),
                'retail_sales_pc': round(retail_sales_pc, 2),
                'education_years': round(education_years, 1),
                'population_density': round(population_density, 0),
                'treated': 1 if province == TREATED_PROVINCE else 0,
                'post': 1 if year >= treatment_year else 0
            })
    
    df = pd.DataFrame(data_list)
    
    return df


def save_simulated_data(df: pd.DataFrame, filename: str = "province_panel.csv"):
    """保存模拟数据"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / filename
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"数据已保存至: {filepath}")
    return filepath


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
    """数据准备主函数"""
    print("=" * 60)
    print("浙江共同富裕示范区政策效应评估 - 数据准备")
    print("=" * 60)
    
    # 1. 打印数据获取指南
    print("\n【步骤1】真实数据获取指南")
    print_data_guide()
    
    # 2. 生成模拟数据
    print("\n【步骤2】生成模拟数据（用于方法演示）")
    df = generate_simulated_data(
        start_year=2010,
        end_year=2023,
        treatment_year=2021,
        treatment_effect=-0.08  # 假设政策使城乡收入比下降8%
    )
    
    # 3. 数据预处理
    print("\n【步骤3】数据预处理")
    df = preprocess_data(df)
    
    # 4. 保存数据
    print("\n【步骤4】保存数据")
    save_simulated_data(df)
    
    # 5. 数据概览
    print("\n【步骤5】数据概览")
    print(f"样本量: {len(df)}")
    print(f"省份数: {df['province'].nunique()}")
    print(f"时间跨度: {df['year'].min()}-{df['year'].max()}")
    print(f"\n处理组: {TREATED_PROVINCE}")
    print(f"处理年份: {TREATMENT_YEAR}")
    print(f"控制组省份数: {len(CONTROL_PROVINCES)}")
    
    print("\n描述性统计:")
    summary = get_summary_statistics(df)
    print(summary)
    
    # 6. 浙江数据预览
    print("\n浙江省数据预览:")
    zj_data = df[df['province'] == '浙江'][['year', 'urban_rural_income_ratio', 'gdp_per_capita', 'urbanization_rate']]
    print(zj_data.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("数据准备完成！")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    main()
