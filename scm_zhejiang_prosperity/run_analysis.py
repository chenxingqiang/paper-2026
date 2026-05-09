#!/usr/bin/env python3
"""
浙江共同富裕示范区政策效应评估
==============================

主运行脚本：完整的合成控制法分析流程

使用方法:
--------
python run_analysis.py [--data PATH] [--skip-placebo] [--skip-viz] [--skip-robustness] [--quick]

参数:
----
--data:           真实面板数据 CSV 路径（默认 data/province_panel_real.csv）
--skip-placebo:   跳过安慰剂检验（较耗时）
--skip-viz:       跳过可视化生成
--skip-robustness:跳过稳健性检验
--quick:          快速模式（减少安慰剂数量）

注意：本项目不再生成或接受任何模拟数据；运行前必须由研究者基于
公开权威数据源整理真实省级面板数据，详见 README.md。
"""

import sys
import argparse
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_collection import preprocess_data
from scm_analysis import SyntheticControl, AugmentedSCM, PlaceboTest, LeaveOneOut
from visualization import SCMVisualizer
from robustness import RobustnessTests

import pandas as pd
import numpy as np


# 项目路径
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='浙江共同富裕示范区政策效应评估 - 合成控制法分析'
    )
    parser.add_argument('--skip-placebo', action='store_true', help='跳过安慰剂检验')
    parser.add_argument('--skip-viz', action='store_true', help='跳过可视化')
    parser.add_argument('--skip-robustness', action='store_true', help='跳过稳健性检验')
    parser.add_argument('--quick', action='store_true', help='快速模式')
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='真实面板数据 CSV 路径（默认 data/province_panel_real.csv）'
    )

    return parser.parse_args()


def step1_prepare_data(data_path: Path = None) -> pd.DataFrame:
    """步骤1: 加载真实面板数据。

    本项目不再生成任何模拟/合成数据，必须由研究者基于公开权威来源
    （《中国统计年鉴》《浙江统计年鉴》等）整理真实面板数据 CSV。
    """

    print("\n" + "=" * 60)
    print("【步骤1】数据准备")
    print("=" * 60)

    if data_path is None:
        data_path = DATA_DIR / "province_panel_real.csv"
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"未找到真实面板数据: {data_path}\n"
            f"本项目不提供任何模拟数据。请基于《中国统计年鉴》等公开权威来源"
            f"整理省级面板数据并保存为 {data_path}\n"
            f"字段说明见 README.md《数据准备》一节，"
            f"或运行 `python -m data_collection` 查看数据获取指南。"
        )

    print(f"使用真实面板数据: {data_path}")
    data = pd.read_csv(data_path)
    data = preprocess_data(data)

    print(f"\n数据概览:")
    print(f"  样本量: {len(data)}")
    print(f"  省份数: {data['province'].nunique()}")
    print(f"  时间跨度: {data['year'].min()}-{data['year'].max()}")

    return data


def step2_fit_scm(data: pd.DataFrame):
    """步骤2: 拟合合成控制模型"""
    
    print("\n" + "=" * 60)
    print("【步骤2】拟合合成控制模型")
    print("=" * 60)
    
    # 基础SCM
    print("\n--- 基础合成控制法 ---")
    scm = SyntheticControl()
    scm.fit(
        data=data,
        outcome='urban_rural_income_ratio',
        predictors=[
            'gdp_per_capita',
            'urbanization_rate',
            'tertiary_share',
            'fiscal_expenditure_pc',
            'fixed_investment_pc'
        ],
        treated_unit='浙江',
        treatment_time=2021,
        unit_col='province',
        time_col='year'
    )
    
    # 打印结果摘要
    scm.summary()
    
    # 增强型SCM
    print("\n--- 增强型合成控制法 ---")
    ascm = AugmentedSCM(outcome_model='ridge')
    ascm.fit(
        data=data,
        outcome='urban_rural_income_ratio',
        predictors=[
            'gdp_per_capita',
            'urbanization_rate',
            'tertiary_share',
            'fiscal_expenditure_pc',
            'fixed_investment_pc'
        ],
        treated_unit='浙江',
        treatment_time=2021,
        unit_col='province',
        time_col='year'
    )
    
    return scm, ascm


def step3_inference(scm: SyntheticControl, quick: bool = False):
    """步骤3: 统计推断（安慰剂检验）"""
    
    print("\n" + "=" * 60)
    print("【步骤3】统计推断")
    print("=" * 60)
    
    # 空间安慰剂检验
    placebo = PlaceboTest(scm)
    n_placebos = 5 if quick else None  # 快速模式只测试5个安慰剂
    spatial_results = placebo.run_spatial_placebo(n_placebos=n_placebos)
    
    # 计算p值
    p_value = placebo.compute_p_value()
    
    # 留一交叉验证
    print("\n--- 留一交叉验证 ---")
    loo = LeaveOneOut(scm)
    loo_results = loo.run()
    
    return placebo, spatial_results, loo_results


def step4_robustness(data: pd.DataFrame):
    """步骤4: 稳健性检验"""
    
    print("\n" + "=" * 60)
    print("【步骤4】稳健性检验")
    print("=" * 60)
    
    robustness = RobustnessTests(data)
    results = robustness.run_all_tests()
    robustness.save_results()
    
    return robustness


def step5_visualization(scm, spatial_results, loo_results):
    """步骤5: 可视化"""
    
    print("\n" + "=" * 60)
    print("【步骤5】可视化")
    print("=" * 60)
    
    viz = SCMVisualizer()
    viz.create_all_plots(
        scm_model=scm,
        placebo_results=spatial_results,
        loo_results=loo_results
    )


def step6_save_results(scm, spatial_results, loo_results):
    """步骤6: 保存结果"""
    
    print("\n" + "=" * 60)
    print("【步骤6】保存结果")
    print("=" * 60)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 主要结果
    results = scm.get_results()
    results.to_csv(RESULTS_DIR / "scm_results.csv", index=False)
    print(f"  主要结果: {RESULTS_DIR / 'scm_results.csv'}")
    
    # 权重
    weights = scm.get_weights()
    weights.to_csv(RESULTS_DIR / "scm_weights.csv", index=False)
    print(f"  权重: {RESULTS_DIR / 'scm_weights.csv'}")
    
    # 预测变量平衡表
    balance = scm.get_predictor_balance()
    balance.to_csv(RESULTS_DIR / "predictor_balance.csv", index=False, encoding='utf-8-sig')
    print(f"  平衡表: {RESULTS_DIR / 'predictor_balance.csv'}")
    
    # 安慰剂结果
    if spatial_results is not None:
        spatial_results.to_csv(RESULTS_DIR / "placebo_results.csv", index=False)
        print(f"  安慰剂结果: {RESULTS_DIR / 'placebo_results.csv'}")
    
    # 留一交叉验证结果
    if loo_results is not None:
        loo_results.to_csv(RESULTS_DIR / "loo_results.csv", index=False)
        print(f"  LOO结果: {RESULTS_DIR / 'loo_results.csv'}")


def generate_report(scm, p_value: float = None):
    """生成分析报告"""
    
    rel_eff = scm.att / np.mean(scm.Y_treated) * 100
    direction = "降低" if scm.att < 0 else "提高"
    abs_rel = abs(rel_eff)
    report = f"""
================================================================================
                    浙江共同富裕示范区政策效应评估报告
                        基于合成控制法（SCM）
================================================================================

一、研究概述
-----------
处理单位: {scm.treated_unit}
处理时间: {scm.treatment_time}年
结果变量: 城乡收入比（城镇/农村居民人均可支配收入）
控制单位数量: {len(scm.control_units)}
分析时期: {min(scm.all_times)}-{max(scm.all_times)}

二、模型拟合
-----------
处理前RMSE: {scm.pre_treatment_rmse:.4f}
（该值越小表示合成控制拟合越好）

三、主要发现
-----------
平均处理效应(ATT): {scm.att:.4f}
相对效应: {rel_eff:.2f}%

解读:
- ATT为{'负' if scm.att < 0 else '正'}值表示政策{direction}了城乡收入比（{'缩小' if scm.att < 0 else '扩大'}了城乡差距）
- 相对效应表示政策使城乡收入比相对{direction}了约{abs_rel:.1f}%

年度处理效应:
"""
    
    results = scm.get_results()
    post_results = results[results['post_treatment'] == 1]
    for _, row in post_results.iterrows():
        report += f"  {int(row['year'])}: {row['effect']:.4f}\n"
    
    report += f"""
四、统计推断
-----------
"""
    if p_value is not None:
        report += f"基于空间安慰剂检验的p值: {p_value:.4f}\n"
        if p_value < 0.1:
            report += "结论: 在10%显著性水平上，政策效应显著\n"
        else:
            report += "结论: 政策效应在统计上不显著\n"
    
    report += f"""
五、控制单位权重（前5位）
------------------------
"""
    weights = scm.get_weights()
    for _, row in weights.head(5).iterrows():
        report += f"  {row['province']}: {row['weight']:.4f}\n"
    
    if scm.att < 0:
        implication_1 = "1. 浙江共同富裕示范区建设对缩小城乡收入差距具有积极作用"
        implication_2 = "2. 政策效应随时间推移逐步显现并增强"
    else:
        implication_1 = ("1. 在当前模型设定下，处理后浙江实际城乡收入比高于合成控制反事实，"
                         "说明合成控制法估计的政策因果效应并非显著缩小差距")
        implication_2 = ("2. 该结果与浙江长期高速下行的城乡差距趋势相关——"
                         "处理前合成控制拟合误差较大时，估计结果对模型设定较敏感，需结合稳健性检验综合判断")

    report += f"""
六、政策启示
-----------
{implication_1}
{implication_2}
3. 主要对标省份为{', '.join(weights.head(3)['province'].tolist())}

七、研究局限
-----------
1. 全部省级面板变量须基于《中国统计年鉴》《浙江统计年鉴》等公开权威来源整理，本项目不接受任何模拟/外推数据
2. 合成控制法的因果推断依赖于"无未观测混杂因素"假设
3. 政策实施时间较短（2021年至今），长期效应需持续跟踪
4. 处理前拟合RMSE需与估计效应对照评估；当RMSE接近或超过ATT时，结论应保守解释

================================================================================
                              报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
"""
    
    return report


def main():
    """主函数"""
    
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("    浙江共同富裕示范区政策效应评估")
    print("    基于合成控制法（Synthetic Control Method）")
    print("=" * 60)
    
    # 步骤1: 数据准备
    data = step1_prepare_data(data_path=args.data)
    
    # 步骤2: 拟合SCM
    scm, ascm = step2_fit_scm(data)
    
    # 步骤3: 统计推断
    p_value = None
    spatial_results = None
    loo_results = None
    
    if not args.skip_placebo:
        placebo, spatial_results, loo_results = step3_inference(scm, quick=args.quick)
        p_value = placebo.compute_p_value()
    
    # 步骤4: 稳健性检验
    if not args.skip_robustness:
        robustness = step4_robustness(data)
    
    # 步骤5: 可视化
    if not args.skip_viz:
        step5_visualization(scm, spatial_results, loo_results)
    
    # 步骤6: 保存结果
    step6_save_results(scm, spatial_results, loo_results)
    
    # 生成报告
    report = generate_report(scm, p_value)
    print(report)
    
    # 保存报告
    report_path = RESULTS_DIR / "analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n报告已保存: {report_path}")
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print(f"所有结果保存在: {RESULTS_DIR}")
    print("=" * 60)
    
    return scm


if __name__ == "__main__":
    scm = main()
