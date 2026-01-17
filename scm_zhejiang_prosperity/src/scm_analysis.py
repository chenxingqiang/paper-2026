"""
合成控制法（Synthetic Control Method）核心实现
==============================================

实现内容：
1. 基础合成控制法（Abadie et al., 2010）
2. 增强型合成控制法（Augmented SCM）
3. 安慰剂检验（Placebo Tests）
4. 留一交叉验证（Leave-One-Out）

参考文献：
- Abadie, A., Diamond, A., & Hainmueller, J. (2010)
- Abadie, A. (2021). Using synthetic controls. JEL.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union
import warnings
from pathlib import Path

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


# =============================================================================
# 一、基础合成控制法类
# =============================================================================

class SyntheticControl:
    """
    合成控制法基础实现
    
    使用方法：
    ---------
    >>> scm = SyntheticControl()
    >>> scm.fit(data, outcome='urban_rural_income_ratio', 
    ...         predictors=['gdp_per_capita', 'urbanization_rate'],
    ...         treated_unit='浙江', treatment_time=2021)
    >>> results = scm.get_results()
    >>> scm.plot()
    """
    
    def __init__(self, optimization_method: str = 'SLSQP'):
        """
        初始化
        
        参数:
        -----
        optimization_method: 优化方法，可选 'SLSQP', 'trust-constr'
        """
        self.optimization_method = optimization_method
        self.weights_ = None
        self.v_weights_ = None
        self.treated_unit = None
        self.control_units = None
        self.treatment_time = None
        self.outcome_var = None
        self.predictors = None
        self.data = None
        self.fitted = False
        
        # 结果存储
        self.Y_treated = None
        self.Y_synthetic = None
        self.treatment_effect = None
        self.pre_treatment_rmse = None
        
    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        predictors: List[str],
        treated_unit: str,
        treatment_time: int,
        unit_col: str = 'province',
        time_col: str = 'year',
        special_predictors: Optional[Dict] = None
    ) -> 'SyntheticControl':
        """
        拟合合成控制模型
        
        参数:
        -----
        data: 面板数据（长格式）
        outcome: 结果变量名
        predictors: 预测变量列表
        treated_unit: 处理单位名称
        treatment_time: 处理时间
        unit_col: 单位列名
        time_col: 时间列名
        special_predictors: 特殊预测变量（指定时间范围的结果变量均值）
        
        返回:
        -----
        self
        """
        self.data = data.copy()
        self.outcome_var = outcome
        self.predictors = predictors
        self.treated_unit = treated_unit
        self.treatment_time = treatment_time
        self.unit_col = unit_col
        self.time_col = time_col
        
        # 获取控制单位
        all_units = data[unit_col].unique()
        self.control_units = [u for u in all_units if u != treated_unit]
        
        # 获取时间范围
        self.all_times = sorted(data[time_col].unique())
        self.pre_times = [t for t in self.all_times if t < treatment_time]
        self.post_times = [t for t in self.all_times if t >= treatment_time]
        
        print(f"处理单位: {treated_unit}")
        print(f"控制单位数: {len(self.control_units)}")
        print(f"处理前时期: {min(self.pre_times)}-{max(self.pre_times)}")
        print(f"处理后时期: {min(self.post_times)}-{max(self.post_times)}")
        
        # 构建预测变量矩阵
        X0, X1 = self._build_predictor_matrix(special_predictors)
        
        # 构建结果变量矩阵（处理前）
        Z0, Z1 = self._build_outcome_matrix(pre_period=True)
        
        # 优化权重
        self.weights_, self.v_weights_ = self._optimize_weights(X0, X1, Z0, Z1)
        
        # 计算合成控制结果
        self._compute_synthetic_outcome()
        
        # 计算处理效应
        self._compute_treatment_effect()
        
        self.fitted = True
        return self
    
    def _build_predictor_matrix(
        self, 
        special_predictors: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构建预测变量矩阵"""
        
        X0_list = []  # 控制单位
        X1_list = []  # 处理单位
        
        # 处理前时期数据
        pre_data = self.data[self.data[self.time_col] < self.treatment_time]
        
        # 常规预测变量（使用处理前均值）
        for predictor in self.predictors:
            # 控制单位均值
            control_means = pre_data.groupby(self.unit_col)[predictor].mean()
            X0_list.append(control_means[self.control_units].values)
            
            # 处理单位均值
            X1_list.append([pre_data[pre_data[self.unit_col] == self.treated_unit][predictor].mean()])
        
        # 特殊预测变量（指定时间范围的结果变量）
        if special_predictors:
            for time_range, var in special_predictors.items():
                start_time, end_time = time_range
                period_data = self.data[
                    (self.data[self.time_col] >= start_time) & 
                    (self.data[self.time_col] <= end_time)
                ]
                control_means = period_data.groupby(self.unit_col)[var].mean()
                X0_list.append(control_means[self.control_units].values)
                X1_list.append([period_data[period_data[self.unit_col] == self.treated_unit][var].mean()])
        
        X0 = np.array(X0_list)  # K x J (预测变量数 x 控制单位数)
        X1 = np.array(X1_list).flatten()  # K x 1
        
        return X0, X1
    
    def _build_outcome_matrix(
        self, 
        pre_period: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构建结果变量矩阵"""
        
        if pre_period:
            times = self.pre_times
        else:
            times = self.all_times
        
        Z0_list = []
        Z1_list = []
        
        for t in times:
            t_data = self.data[self.data[self.time_col] == t]
            
            # 控制单位
            control_values = t_data.set_index(self.unit_col).loc[self.control_units, self.outcome_var].values
            Z0_list.append(control_values)
            
            # 处理单位
            treated_value = t_data[t_data[self.unit_col] == self.treated_unit][self.outcome_var].values[0]
            Z1_list.append(treated_value)
        
        Z0 = np.array(Z0_list)  # T x J
        Z1 = np.array(Z1_list)  # T x 1
        
        return Z0, Z1
    
    def _optimize_weights(
        self,
        X0: np.ndarray,
        X1: np.ndarray,
        Z0: np.ndarray,
        Z1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        优化合成控制权重
        
        采用嵌套优化：
        - 外层：优化V（预测变量权重）
        - 内层：优化W（控制单位权重）
        """
        
        n_controls = X0.shape[1]
        n_predictors = X0.shape[0]
        
        def inner_optimization(V: np.ndarray) -> Tuple[np.ndarray, float]:
            """内层优化：给定V，优化W"""
            
            # 加权预测变量差异
            V_diag = np.diag(V)
            
            def objective_w(W):
                diff = X1 - X0 @ W
                return diff @ V_diag @ diff
            
            # 约束：权重和为1，非负
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            ]
            bounds = [(0, 1) for _ in range(n_controls)]
            
            # 初始值
            W0 = np.ones(n_controls) / n_controls
            
            result = minimize(
                objective_w, W0,
                method=self.optimization_method,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            return result.x, result.fun
        
        def outer_objective(V: np.ndarray) -> float:
            """外层目标：处理前结果变量的拟合误差"""
            W_opt, _ = inner_optimization(V)
            
            # 合成控制的处理前结果
            Z_synthetic = Z0 @ W_opt
            
            # 均方误差
            mse = np.mean((Z1 - Z_synthetic) ** 2)
            return mse
        
        # 优化V
        V0 = np.ones(n_predictors) / n_predictors
        bounds_v = [(0.0001, 1) for _ in range(n_predictors)]
        
        result_v = minimize(
            outer_objective, V0,
            method='L-BFGS-B',
            bounds=bounds_v,
            options={'maxiter': 500}
        )
        
        V_opt = result_v.x
        V_opt = V_opt / np.sum(V_opt)  # 归一化
        
        # 用最优V计算最优W
        W_opt, _ = inner_optimization(V_opt)
        
        return W_opt, V_opt
    
    def _compute_synthetic_outcome(self):
        """计算合成控制结果"""
        
        # 全时期结果变量矩阵
        Z0_full, Z1_full = self._build_outcome_matrix(pre_period=False)
        
        self.Y_treated = Z1_full
        self.Y_synthetic = Z0_full @ self.weights_
        
    def _compute_treatment_effect(self):
        """计算处理效应"""
        
        self.treatment_effect = self.Y_treated - self.Y_synthetic
        
        # 处理前RMSE
        pre_idx = len(self.pre_times)
        pre_effect = self.treatment_effect[:pre_idx]
        self.pre_treatment_rmse = np.sqrt(np.mean(pre_effect ** 2))
        
        # 处理后平均效应
        post_effect = self.treatment_effect[pre_idx:]
        self.att = np.mean(post_effect)
        
        print(f"\n处理前RMSE: {self.pre_treatment_rmse:.4f}")
        print(f"处理后平均效应(ATT): {self.att:.4f}")
    
    def get_weights(self) -> pd.DataFrame:
        """获取控制单位权重"""
        weights_df = pd.DataFrame({
            'province': self.control_units,
            'weight': self.weights_
        })
        weights_df = weights_df.sort_values('weight', ascending=False)
        weights_df = weights_df[weights_df['weight'] > 0.001]
        return weights_df
    
    def get_predictor_balance(self) -> pd.DataFrame:
        """获取预测变量平衡表"""
        
        pre_data = self.data[self.data[self.time_col] < self.treatment_time]
        
        balance_list = []
        for predictor in self.predictors:
            # 处理单位
            treated_mean = pre_data[pre_data[self.unit_col] == self.treated_unit][predictor].mean()
            
            # 合成控制
            control_data = pre_data[pre_data[self.unit_col].isin(self.control_units)]
            control_means = control_data.groupby(self.unit_col)[predictor].mean()
            synthetic_mean = np.sum(control_means[self.control_units].values * self.weights_)
            
            # 控制组简单平均
            simple_mean = control_means.mean()
            
            balance_list.append({
                '变量': predictor,
                '浙江': round(treated_mean, 3),
                '合成浙江': round(synthetic_mean, 3),
                '控制组均值': round(simple_mean, 3),
                '改善比例(%)': round((1 - abs(treated_mean - synthetic_mean) / abs(treated_mean - simple_mean)) * 100, 1)
            })
        
        return pd.DataFrame(balance_list)
    
    def get_results(self) -> pd.DataFrame:
        """获取完整结果"""
        
        results_df = pd.DataFrame({
            'year': self.all_times,
            'treated': self.Y_treated,
            'synthetic': self.Y_synthetic,
            'effect': self.treatment_effect,
            'post_treatment': [1 if t >= self.treatment_time else 0 for t in self.all_times]
        })
        
        return results_df
    
    def summary(self):
        """打印结果摘要"""
        
        print("\n" + "=" * 60)
        print("合成控制法结果摘要")
        print("=" * 60)
        
        print(f"\n【基本信息】")
        print(f"处理单位: {self.treated_unit}")
        print(f"处理时间: {self.treatment_time}")
        print(f"结果变量: {self.outcome_var}")
        print(f"控制单位数: {len(self.control_units)}")
        
        print(f"\n【模型拟合】")
        print(f"处理前RMSE: {self.pre_treatment_rmse:.4f}")
        
        print(f"\n【处理效应】")
        results = self.get_results()
        post_results = results[results['post_treatment'] == 1]
        
        for _, row in post_results.iterrows():
            print(f"  {int(row['year'])}: 效应 = {row['effect']:.4f} "
                  f"(实际: {row['treated']:.3f}, 合成: {row['synthetic']:.3f})")
        
        print(f"\n  平均处理效应(ATT): {self.att:.4f}")
        print(f"  相对效应: {self.att / np.mean(self.Y_treated) * 100:.2f}%")
        
        print(f"\n【控制单位权重】")
        weights = self.get_weights()
        print(weights.head(10).to_string(index=False))
        
        print(f"\n【预测变量平衡】")
        balance = self.get_predictor_balance()
        print(balance.to_string(index=False))
        
        print("\n" + "=" * 60)


# =============================================================================
# 二、增强型合成控制法
# =============================================================================

class AugmentedSCM(SyntheticControl):
    """
    增强型合成控制法（Augmented Synthetic Control Method）
    
    通过结果回归模型增强估计精度
    参考：Ben-Michael et al. (2021)
    """
    
    def __init__(self, outcome_model: str = 'ridge', **kwargs):
        """
        参数:
        -----
        outcome_model: 结果模型类型，可选 'ridge', 'lasso', 'linear'
        """
        super().__init__(**kwargs)
        self.outcome_model = outcome_model
        self.augmentation_term = None
        
    def fit(self, *args, **kwargs) -> 'AugmentedSCM':
        """拟合增强型SCM"""
        
        # 首先运行标准SCM
        super().fit(*args, **kwargs)
        
        # 计算增强项
        self._compute_augmentation()
        
        return self
    
    def _compute_augmentation(self):
        """计算增强项（偏差校正）"""
        from sklearn.linear_model import Ridge, Lasso, LinearRegression
        
        # 选择模型
        if self.outcome_model == 'ridge':
            model = Ridge(alpha=1.0)
        elif self.outcome_model == 'lasso':
            model = Lasso(alpha=0.1)
        else:
            model = LinearRegression()
        
        # 使用处理前数据训练模型
        pre_data = self.data[self.data[self.time_col] < self.treatment_time]
        
        # 特征：预测变量
        X_train = pre_data[pre_data[self.unit_col].isin(self.control_units)][self.predictors]
        y_train = pre_data[pre_data[self.unit_col].isin(self.control_units)][self.outcome_var]
        
        model.fit(X_train, y_train)
        
        # 预测处理单位的反事实结果
        X_treated = pre_data[pre_data[self.unit_col] == self.treated_unit][self.predictors]
        y_pred = model.predict(X_treated)
        y_actual = pre_data[pre_data[self.unit_col] == self.treated_unit][self.outcome_var].values
        
        # 增强项：实际值与预测值的差异
        self.augmentation_term = np.mean(y_actual - y_pred)
        
        # 调整合成控制结果
        self.Y_synthetic_augmented = self.Y_synthetic + self.augmentation_term
        self.treatment_effect_augmented = self.Y_treated - self.Y_synthetic_augmented
        
        # 更新ATT
        pre_idx = len(self.pre_times)
        self.att_augmented = np.mean(self.treatment_effect_augmented[pre_idx:])
        
        print(f"\n【增强型SCM】")
        print(f"增强项: {self.augmentation_term:.4f}")
        print(f"调整后ATT: {self.att_augmented:.4f}")


# =============================================================================
# 三、安慰剂检验
# =============================================================================

class PlaceboTest:
    """
    安慰剂检验（Placebo Tests）
    
    包括：
    1. 空间安慰剂：将每个控制单位假设为处理单位
    2. 时间安慰剂：假设政策提前实施
    """
    
    def __init__(self, scm_model: SyntheticControl):
        """
        参数:
        -----
        scm_model: 已拟合的SyntheticControl模型
        """
        self.base_model = scm_model
        self.spatial_results = None
        self.temporal_results = None
        
    def run_spatial_placebo(
        self,
        n_placebos: Optional[int] = None,
        pre_rmse_threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        空间安慰剂检验
        
        参数:
        -----
        n_placebos: 安慰剂数量（None表示所有控制单位）
        pre_rmse_threshold: 处理前RMSE阈值（过滤拟合不好的）
        
        返回:
        -----
        DataFrame: 各安慰剂单位的效应
        """
        print("\n运行空间安慰剂检验...")
        
        control_units = self.base_model.control_units
        if n_placebos:
            control_units = control_units[:n_placebos]
        
        results_list = []
        
        for i, placebo_unit in enumerate(control_units):
            print(f"  [{i+1}/{len(control_units)}] 处理单位: {placebo_unit}", end='\r')
            
            try:
                # 新的控制组（排除当前安慰剂单位）
                new_controls = [u for u in self.base_model.control_units if u != placebo_unit]
                new_controls.append(self.base_model.treated_unit)  # 真实处理单位加入控制组
                
                # 筛选数据
                placebo_data = self.base_model.data[
                    self.base_model.data[self.base_model.unit_col].isin(new_controls + [placebo_unit])
                ]
                
                # 拟合SCM
                placebo_scm = SyntheticControl()
                placebo_scm.fit(
                    data=placebo_data,
                    outcome=self.base_model.outcome_var,
                    predictors=self.base_model.predictors,
                    treated_unit=placebo_unit,
                    treatment_time=self.base_model.treatment_time,
                    unit_col=self.base_model.unit_col,
                    time_col=self.base_model.time_col
                )
                
                # 过滤拟合不好的
                if placebo_scm.pre_treatment_rmse > pre_rmse_threshold * self.base_model.pre_treatment_rmse:
                    continue
                
                # 存储结果
                for t, effect in zip(self.base_model.all_times, placebo_scm.treatment_effect):
                    results_list.append({
                        'unit': placebo_unit,
                        'year': t,
                        'effect': effect,
                        'pre_rmse': placebo_scm.pre_treatment_rmse,
                        'is_treated': False
                    })
                    
            except Exception as e:
                warnings.warn(f"安慰剂单位 {placebo_unit} 失败: {e}")
                continue
        
        # 添加真实处理单位结果
        for t, effect in zip(self.base_model.all_times, self.base_model.treatment_effect):
            results_list.append({
                'unit': self.base_model.treated_unit,
                'year': t,
                'effect': effect,
                'pre_rmse': self.base_model.pre_treatment_rmse,
                'is_treated': True
            })
        
        self.spatial_results = pd.DataFrame(results_list)
        print(f"\n空间安慰剂检验完成，共 {self.spatial_results['unit'].nunique()} 个单位")
        
        return self.spatial_results
    
    def run_temporal_placebo(
        self,
        placebo_times: List[int]
    ) -> pd.DataFrame:
        """
        时间安慰剂检验（假设政策提前实施）
        
        参数:
        -----
        placebo_times: 安慰剂处理时间列表
        
        返回:
        -----
        DataFrame: 各时间点的效应
        """
        print("\n运行时间安慰剂检验...")
        
        results_list = []
        
        for placebo_time in placebo_times:
            print(f"  假设处理时间: {placebo_time}")
            
            try:
                # 仅使用安慰剂时间之前的数据
                placebo_data = self.base_model.data[
                    self.base_model.data[self.base_model.time_col] < self.base_model.treatment_time
                ]
                
                # 拟合SCM
                placebo_scm = SyntheticControl()
                placebo_scm.fit(
                    data=placebo_data,
                    outcome=self.base_model.outcome_var,
                    predictors=self.base_model.predictors,
                    treated_unit=self.base_model.treated_unit,
                    treatment_time=placebo_time,
                    unit_col=self.base_model.unit_col,
                    time_col=self.base_model.time_col
                )
                
                # 存储结果
                for t, effect in zip(placebo_scm.all_times, placebo_scm.treatment_effect):
                    results_list.append({
                        'placebo_time': placebo_time,
                        'year': t,
                        'effect': effect,
                        'pre_rmse': placebo_scm.pre_treatment_rmse
                    })
                    
            except Exception as e:
                warnings.warn(f"时间安慰剂 {placebo_time} 失败: {e}")
                continue
        
        self.temporal_results = pd.DataFrame(results_list)
        print(f"时间安慰剂检验完成")
        
        return self.temporal_results
    
    def compute_p_value(self) -> float:
        """
        计算p值（基于空间安慰剂检验）
        
        p值 = 安慰剂效应 >= 实际效应的比例
        """
        if self.spatial_results is None:
            raise ValueError("请先运行空间安慰剂检验")
        
        # 处理后的效应
        post_results = self.spatial_results[
            self.spatial_results['year'] >= self.base_model.treatment_time
        ]
        
        # 真实处理效应
        treated_effect = post_results[post_results['is_treated']]['effect'].mean()
        
        # 安慰剂效应
        placebo_effects = post_results[~post_results['is_treated']].groupby('unit')['effect'].mean()
        
        # p值（双尾）
        p_value = np.mean(np.abs(placebo_effects) >= np.abs(treated_effect))
        
        print(f"\n【推断统计】")
        print(f"真实处理效应: {treated_effect:.4f}")
        print(f"安慰剂效应均值: {placebo_effects.mean():.4f}")
        print(f"安慰剂效应标准差: {placebo_effects.std():.4f}")
        print(f"p值: {p_value:.4f}")
        
        return p_value


# =============================================================================
# 四、留一交叉验证
# =============================================================================

class LeaveOneOut:
    """留一交叉验证（Leave-One-Out）"""
    
    def __init__(self, scm_model: SyntheticControl):
        self.base_model = scm_model
        self.loo_results = None
        
    def run(self) -> pd.DataFrame:
        """运行留一交叉验证"""
        print("\n运行留一交叉验证...")
        
        # 获取权重非零的控制单位
        weights_df = self.base_model.get_weights()
        important_units = weights_df[weights_df['weight'] > 0.05]['province'].tolist()
        
        results_list = []
        
        for excluded_unit in important_units:
            print(f"  排除: {excluded_unit}")
            
            try:
                # 排除该单位
                new_controls = [u for u in self.base_model.control_units if u != excluded_unit]
                loo_data = self.base_model.data[
                    self.base_model.data[self.base_model.unit_col].isin(
                        new_controls + [self.base_model.treated_unit]
                    )
                ]
                
                # 拟合SCM
                loo_scm = SyntheticControl()
                loo_scm.fit(
                    data=loo_data,
                    outcome=self.base_model.outcome_var,
                    predictors=self.base_model.predictors,
                    treated_unit=self.base_model.treated_unit,
                    treatment_time=self.base_model.treatment_time,
                    unit_col=self.base_model.unit_col,
                    time_col=self.base_model.time_col
                )
                
                # 存储结果
                for t, y_syn in zip(self.base_model.all_times, loo_scm.Y_synthetic):
                    results_list.append({
                        'excluded_unit': excluded_unit,
                        'year': t,
                        'synthetic': y_syn,
                        'att': loo_scm.att
                    })
                    
            except Exception as e:
                warnings.warn(f"排除 {excluded_unit} 失败: {e}")
                continue
        
        self.loo_results = pd.DataFrame(results_list)
        
        # 打印摘要
        att_values = self.loo_results.groupby('excluded_unit')['att'].first()
        print(f"\n【留一交叉验证结果】")
        print(f"ATT范围: [{att_values.min():.4f}, {att_values.max():.4f}]")
        print(f"原始ATT: {self.base_model.att:.4f}")
        
        return self.loo_results


# =============================================================================
# 五、主函数
# =============================================================================

def main():
    """主分析函数"""
    
    print("=" * 60)
    print("浙江共同富裕示范区政策效应评估")
    print("基于合成控制法（SCM）")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n【步骤1】加载数据")
    data_path = DATA_DIR / "province_panel.csv"
    
    if not data_path.exists():
        print("数据文件不存在，生成模拟数据...")
        from data_collection import generate_simulated_data, save_simulated_data
        data = generate_simulated_data()
        save_simulated_data(data)
    
    data = pd.read_csv(data_path)
    print(f"数据加载完成: {len(data)} 行")
    
    # 2. 拟合SCM模型
    print("\n【步骤2】拟合合成控制模型")
    
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
    
    # 3. 输出结果
    print("\n【步骤3】结果摘要")
    scm.summary()
    
    # 4. 保存结果
    print("\n【步骤4】保存结果")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = scm.get_results()
    results.to_csv(RESULTS_DIR / "scm_results.csv", index=False)
    
    weights = scm.get_weights()
    weights.to_csv(RESULTS_DIR / "scm_weights.csv", index=False)
    
    balance = scm.get_predictor_balance()
    balance.to_csv(RESULTS_DIR / "predictor_balance.csv", index=False, encoding='utf-8-sig')
    
    print("结果已保存至 results/ 目录")
    
    # 5. 安慰剂检验
    print("\n【步骤5】安慰剂检验")
    placebo = PlaceboTest(scm)
    spatial_results = placebo.run_spatial_placebo(n_placebos=10)
    spatial_results.to_csv(RESULTS_DIR / "placebo_results.csv", index=False)
    
    p_value = placebo.compute_p_value()
    
    # 6. 留一交叉验证
    print("\n【步骤6】留一交叉验证")
    loo = LeaveOneOut(scm)
    loo_results = loo.run()
    loo_results.to_csv(RESULTS_DIR / "loo_results.csv", index=False)
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    
    return scm, placebo, loo


if __name__ == "__main__":
    scm, placebo, loo = main()
