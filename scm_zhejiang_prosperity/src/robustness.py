"""
稳健性检验模块
==============

提供多种稳健性检验方法：
1. 不同结果变量
2. 不同预测变量组合
3. 不同处理时间
4. 排除特定省份
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import warnings

from scm_analysis import SyntheticControl

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


class RobustnessTests:
    """稳健性检验类"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化
        
        参数:
        -----
        data: 面板数据
        """
        self.data = data
        self.results = {}
        
    def test_different_outcomes(
        self,
        outcomes: List[str],
        predictors: List[str],
        treated_unit: str,
        treatment_time: int
    ) -> pd.DataFrame:
        """
        测试不同结果变量
        
        参数:
        -----
        outcomes: 结果变量列表
        predictors: 预测变量列表
        treated_unit: 处理单位
        treatment_time: 处理时间
        
        返回:
        -----
        DataFrame: 各结果变量的ATT
        """
        print("\n【稳健性检验1】不同结果变量")
        print("-" * 40)
        
        results_list = []
        
        for outcome in outcomes:
            print(f"  测试结果变量: {outcome}")
            
            try:
                scm = SyntheticControl()
                scm.fit(
                    data=self.data,
                    outcome=outcome,
                    predictors=[p for p in predictors if p != outcome],
                    treated_unit=treated_unit,
                    treatment_time=treatment_time
                )
                
                results_list.append({
                    'outcome': outcome,
                    'ATT': scm.att,
                    'pre_RMSE': scm.pre_treatment_rmse,
                    'relative_effect': scm.att / np.mean(scm.Y_treated) * 100
                })
                
            except Exception as e:
                warnings.warn(f"结果变量 {outcome} 失败: {e}")
        
        results_df = pd.DataFrame(results_list)
        self.results['different_outcomes'] = results_df
        
        print("\n结果汇总:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def test_different_predictors(
        self,
        predictor_sets: Dict[str, List[str]],
        outcome: str,
        treated_unit: str,
        treatment_time: int
    ) -> pd.DataFrame:
        """
        测试不同预测变量组合
        
        参数:
        -----
        predictor_sets: 预测变量组合字典 {'组合名': [变量列表]}
        outcome: 结果变量
        treated_unit: 处理单位
        treatment_time: 处理时间
        
        返回:
        -----
        DataFrame: 各组合的ATT
        """
        print("\n【稳健性检验2】不同预测变量组合")
        print("-" * 40)
        
        results_list = []
        
        for name, predictors in predictor_sets.items():
            print(f"  测试组合: {name}")
            
            try:
                scm = SyntheticControl()
                scm.fit(
                    data=self.data,
                    outcome=outcome,
                    predictors=predictors,
                    treated_unit=treated_unit,
                    treatment_time=treatment_time
                )
                
                results_list.append({
                    'predictor_set': name,
                    'n_predictors': len(predictors),
                    'ATT': scm.att,
                    'pre_RMSE': scm.pre_treatment_rmse
                })
                
            except Exception as e:
                warnings.warn(f"预测变量组合 {name} 失败: {e}")
        
        results_df = pd.DataFrame(results_list)
        self.results['different_predictors'] = results_df
        
        print("\n结果汇总:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def test_different_treatment_times(
        self,
        treatment_times: List[int],
        outcome: str,
        predictors: List[str],
        treated_unit: str,
        actual_treatment_time: int
    ) -> pd.DataFrame:
        """
        测试不同处理时间（时间安慰剂）
        
        参数:
        -----
        treatment_times: 假设的处理时间列表
        outcome: 结果变量
        predictors: 预测变量
        treated_unit: 处理单位
        actual_treatment_time: 实际处理时间
        
        返回:
        -----
        DataFrame: 各时间点的ATT
        """
        print("\n【稳健性检验3】不同处理时间（时间安慰剂）")
        print("-" * 40)
        
        results_list = []
        
        for t_time in treatment_times:
            print(f"  假设处理时间: {t_time}")
            
            # 只使用实际处理前的数据
            test_data = self.data[self.data['year'] < actual_treatment_time]
            
            try:
                scm = SyntheticControl()
                scm.fit(
                    data=test_data,
                    outcome=outcome,
                    predictors=predictors,
                    treated_unit=treated_unit,
                    treatment_time=t_time
                )
                
                results_list.append({
                    'treatment_time': t_time,
                    'is_actual': t_time == actual_treatment_time,
                    'ATT': scm.att,
                    'pre_RMSE': scm.pre_treatment_rmse
                })
                
            except Exception as e:
                warnings.warn(f"处理时间 {t_time} 失败: {e}")
        
        results_df = pd.DataFrame(results_list)
        self.results['different_times'] = results_df
        
        print("\n结果汇总:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def test_exclude_provinces(
        self,
        provinces_to_exclude: List[str],
        outcome: str,
        predictors: List[str],
        treated_unit: str,
        treatment_time: int
    ) -> pd.DataFrame:
        """
        测试排除特定省份
        
        参数:
        -----
        provinces_to_exclude: 要排除的省份列表
        outcome: 结果变量
        predictors: 预测变量
        treated_unit: 处理单位
        treatment_time: 处理时间
        
        返回:
        -----
        DataFrame: 排除各省份后的ATT
        """
        print("\n【稳健性检验4】排除特定省份")
        print("-" * 40)
        
        results_list = []
        
        # 基准（不排除任何省份）
        print("  基准模型（全样本）")
        scm_base = SyntheticControl()
        scm_base.fit(
            data=self.data,
            outcome=outcome,
            predictors=predictors,
            treated_unit=treated_unit,
            treatment_time=treatment_time
        )
        
        results_list.append({
            'excluded': 'None (Baseline)',
            'ATT': scm_base.att,
            'pre_RMSE': scm_base.pre_treatment_rmse,
            'n_controls': len(scm_base.control_units)
        })
        
        # 逐个排除
        for province in provinces_to_exclude:
            print(f"  排除: {province}")
            
            test_data = self.data[self.data['province'] != province]
            
            try:
                scm = SyntheticControl()
                scm.fit(
                    data=test_data,
                    outcome=outcome,
                    predictors=predictors,
                    treated_unit=treated_unit,
                    treatment_time=treatment_time
                )
                
                results_list.append({
                    'excluded': province,
                    'ATT': scm.att,
                    'pre_RMSE': scm.pre_treatment_rmse,
                    'n_controls': len(scm.control_units)
                })
                
            except Exception as e:
                warnings.warn(f"排除 {province} 失败: {e}")
        
        results_df = pd.DataFrame(results_list)
        self.results['exclude_provinces'] = results_df
        
        print("\n结果汇总:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def run_all_tests(
        self,
        outcome: str = 'urban_rural_income_ratio',
        predictors: List[str] = None,
        treated_unit: str = '浙江',
        treatment_time: int = 2021
    ) -> Dict:
        """运行所有稳健性检验"""
        
        if predictors is None:
            predictors = [
                'gdp_per_capita',
                'urbanization_rate',
                'tertiary_share',
                'fiscal_expenditure_pc',
                'fixed_investment_pc'
            ]
        
        print("\n" + "=" * 60)
        print("开始稳健性检验")
        print("=" * 60)
        
        # 1. 不同结果变量
        self.test_different_outcomes(
            outcomes=['urban_rural_income_ratio', 'gdp_per_capita', 'rural_income'],
            predictors=predictors,
            treated_unit=treated_unit,
            treatment_time=treatment_time
        )
        
        # 2. 不同预测变量组合
        self.test_different_predictors(
            predictor_sets={
                '完整模型': predictors,
                '精简模型': ['gdp_per_capita', 'urbanization_rate'],
                '扩展模型': predictors + ['education_years', 'retail_sales_pc']
            },
            outcome=outcome,
            treated_unit=treated_unit,
            treatment_time=treatment_time
        )
        
        # 3. 不同处理时间
        self.test_different_treatment_times(
            treatment_times=[2017, 2018, 2019, 2020],
            outcome=outcome,
            predictors=predictors,
            treated_unit=treated_unit,
            actual_treatment_time=treatment_time
        )
        
        # 4. 排除特定省份（排除权重较大的省份）
        self.test_exclude_provinces(
            provinces_to_exclude=['江苏', '广东', '山东', '福建'],
            outcome=outcome,
            predictors=predictors,
            treated_unit=treated_unit,
            treatment_time=treatment_time
        )
        
        print("\n" + "=" * 60)
        print("稳健性检验完成！")
        print("=" * 60)
        
        return self.results
    
    def save_results(self, output_dir: Optional[Path] = None):
        """保存稳健性检验结果"""
        
        if output_dir is None:
            output_dir = RESULTS_DIR / "robustness"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df in self.results.items():
            filepath = output_dir / f"robustness_{name}.csv"
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"已保存: {filepath}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    """稳健性检验主函数"""
    
    print("=" * 60)
    print("合成控制法稳健性检验")
    print("=" * 60)
    
    # 加载数据
    data_path = DATA_DIR / "province_panel_real.csv"

    if not data_path.exists():
        print(
            f"未找到真实面板数据: {data_path}\n"
            f"本项目不提供任何模拟数据，请先准备真实数据 CSV，"
            f"详见 README.md《数据准备》一节。"
        )
        return
    
    data = pd.read_csv(data_path)
    
    # 运行稳健性检验
    robustness = RobustnessTests(data)
    results = robustness.run_all_tests()
    
    # 保存结果
    robustness.save_results()
    
    print("\n稳健性检验结果已保存！")


if __name__ == "__main__":
    main()
