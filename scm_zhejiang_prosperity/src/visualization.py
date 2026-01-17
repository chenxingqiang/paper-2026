"""
可视化模块
==========

提供合成控制法结果的可视化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from typing import Optional, List
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


class SCMVisualizer:
    """合成控制法可视化器"""
    
    def __init__(self, scm_model=None, style: str = 'seaborn-v0_8-whitegrid'):
        """
        初始化
        
        参数:
        -----
        scm_model: SyntheticControl模型实例
        style: matplotlib样式
        """
        self.scm = scm_model
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-whitegrid')
        
        # 创建输出目录
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        
        # 颜色方案
        self.colors = {
            'treated': '#E74C3C',      # 红色 - 实际值
            'synthetic': '#3498DB',    # 蓝色 - 合成值
            'effect': '#2ECC71',       # 绿色 - 效应
            'placebo': '#BDC3C7',      # 灰色 - 安慰剂
            'treatment_line': '#95A5A6' # 处理时间线
        }
    
    def plot_trends(
        self, 
        results: Optional[pd.DataFrame] = None,
        treatment_time: Optional[int] = None,
        title: str = 'Synthetic Control Method Results',
        outcome_label: str = 'Urban-Rural Income Ratio',
        save_path: Optional[str] = None,
        figsize: tuple = (12, 6)
    ):
        """
        绘制趋势对比图（实际值 vs 合成值）
        
        参数:
        -----
        results: 结果DataFrame（包含year, treated, synthetic列）
        treatment_time: 处理时间
        title: 图表标题
        outcome_label: Y轴标签
        save_path: 保存路径
        figsize: 图表大小
        """
        if results is None and self.scm is not None:
            results = self.scm.get_results()
            treatment_time = self.scm.treatment_time
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制实际值
        ax.plot(
            results['year'], results['treated'],
            color=self.colors['treated'],
            linewidth=2.5,
            marker='o',
            markersize=6,
            label='Zhejiang (Actual)'
        )
        
        # 绘制合成值
        ax.plot(
            results['year'], results['synthetic'],
            color=self.colors['synthetic'],
            linewidth=2.5,
            linestyle='--',
            marker='s',
            markersize=6,
            label='Synthetic Zhejiang'
        )
        
        # 绘制处理时间线
        ax.axvline(
            x=treatment_time,
            color=self.colors['treatment_line'],
            linestyle=':',
            linewidth=2,
            label=f'Treatment ({treatment_time})'
        )
        
        # 填充处理后区域
        post_mask = results['year'] >= treatment_time
        ax.fill_between(
            results[post_mask]['year'],
            results[post_mask]['treated'],
            results[post_mask]['synthetic'],
            alpha=0.2,
            color=self.colors['effect'],
            label='Treatment Effect'
        )
        
        # 设置标签和标题
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(outcome_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 图例
        ax.legend(loc='best', frameon=True, fontsize=10)
        
        # 网格
        ax.grid(True, alpha=0.3)
        
        # 设置x轴刻度
        ax.set_xticks(results['year'].unique()[::2])
        
        plt.tight_layout()
        
        # 保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        else:
            plt.savefig(FIGURES_DIR / "scm_trends.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig, ax
    
    def plot_treatment_effect(
        self,
        results: Optional[pd.DataFrame] = None,
        treatment_time: Optional[int] = None,
        title: str = 'Treatment Effect Over Time',
        save_path: Optional[str] = None,
        figsize: tuple = (12, 5)
    ):
        """
        绘制处理效应图
        """
        if results is None and self.scm is not None:
            results = self.scm.get_results()
            treatment_time = self.scm.treatment_time
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 处理前和处理后分开绘制
        pre_mask = results['year'] < treatment_time
        post_mask = results['year'] >= treatment_time
        
        # 处理前效应
        ax.bar(
            results[pre_mask]['year'],
            results[pre_mask]['effect'],
            color='gray',
            alpha=0.5,
            label='Pre-treatment Gap'
        )
        
        # 处理后效应
        colors = [self.colors['effect'] if e < 0 else '#E74C3C' 
                  for e in results[post_mask]['effect']]
        ax.bar(
            results[post_mask]['year'],
            results[post_mask]['effect'],
            color=colors,
            alpha=0.8,
            label='Treatment Effect'
        )
        
        # 零线
        ax.axhline(y=0, color='black', linewidth=1)
        
        # 处理时间线
        ax.axvline(
            x=treatment_time - 0.5,
            color=self.colors['treatment_line'],
            linestyle='--',
            linewidth=2,
            label=f'Treatment ({treatment_time})'
        )
        
        # 标签
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Effect (Treated - Synthetic)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(FIGURES_DIR / "treatment_effect.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig, ax
    
    def plot_weights(
        self,
        weights: Optional[pd.DataFrame] = None,
        top_n: int = 10,
        title: str = 'Control Unit Weights',
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6)
    ):
        """
        绘制权重分布图
        """
        if weights is None and self.scm is not None:
            weights = self.scm.get_weights()
        
        # 取top_n
        weights_plot = weights.head(top_n).copy()
        weights_plot = weights_plot.sort_values('weight')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 水平条形图
        bars = ax.barh(
            weights_plot['province'],
            weights_plot['weight'],
            color=self.colors['synthetic'],
            alpha=0.8
        )
        
        # 在条形上添加数值
        for bar, weight in zip(bars, weights_plot['weight']):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{weight:.3f}',
                va='center',
                fontsize=10
            )
        
        ax.set_xlabel('Weight', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(weights_plot['weight']) * 1.2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(FIGURES_DIR / "weights.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig, ax
    
    def plot_placebo(
        self,
        placebo_results: pd.DataFrame,
        treatment_time: int,
        title: str = 'Placebo Test Results',
        save_path: Optional[str] = None,
        figsize: tuple = (12, 6)
    ):
        """
        绘制安慰剂检验图
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制所有安慰剂单位
        for unit in placebo_results['unit'].unique():
            unit_data = placebo_results[placebo_results['unit'] == unit]
            is_treated = unit_data['is_treated'].iloc[0]
            
            if is_treated:
                ax.plot(
                    unit_data['year'],
                    unit_data['effect'],
                    color=self.colors['treated'],
                    linewidth=3,
                    label='Zhejiang (Treated)',
                    zorder=10
                )
            else:
                ax.plot(
                    unit_data['year'],
                    unit_data['effect'],
                    color=self.colors['placebo'],
                    linewidth=1,
                    alpha=0.5
                )
        
        # 处理时间线
        ax.axvline(
            x=treatment_time,
            color='black',
            linestyle='--',
            linewidth=1.5
        )
        
        # 零线
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        # 添加"Placebo units"标签（只添加一次）
        ax.plot([], [], color=self.colors['placebo'], linewidth=1, label='Placebo Units')
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Effect (Treated - Synthetic)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(FIGURES_DIR / "placebo_test.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig, ax
    
    def plot_ratio_distribution(
        self,
        placebo_results: pd.DataFrame,
        treatment_time: int,
        title: str = 'Post/Pre RMSPE Ratio Distribution',
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6)
    ):
        """
        绘制RMSPE比率分布图（更严格的推断）
        """
        # 计算各单位的post/pre RMSPE比率
        ratios = []
        
        for unit in placebo_results['unit'].unique():
            unit_data = placebo_results[placebo_results['unit'] == unit]
            
            pre_effects = unit_data[unit_data['year'] < treatment_time]['effect']
            post_effects = unit_data[unit_data['year'] >= treatment_time]['effect']
            
            pre_rmspe = np.sqrt(np.mean(pre_effects ** 2))
            post_rmspe = np.sqrt(np.mean(post_effects ** 2))
            
            if pre_rmspe > 0:
                ratio = post_rmspe / pre_rmspe
                ratios.append({
                    'unit': unit,
                    'ratio': ratio,
                    'is_treated': unit_data['is_treated'].iloc[0]
                })
        
        ratios_df = pd.DataFrame(ratios).sort_values('ratio', ascending=False)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 条形图
        colors = [self.colors['treated'] if is_t else self.colors['placebo'] 
                  for is_t in ratios_df['is_treated']]
        
        bars = ax.barh(
            range(len(ratios_df)),
            ratios_df['ratio'],
            color=colors,
            alpha=0.8
        )
        
        # 标注浙江
        treated_idx = ratios_df[ratios_df['is_treated']].index[0]
        treated_rank = list(ratios_df.index).index(treated_idx) + 1
        
        ax.set_yticks(range(len(ratios_df)))
        ax.set_yticklabels(ratios_df['unit'], fontsize=8)
        ax.set_xlabel('Post/Pre RMSPE Ratio', fontsize=12)
        ax.set_title(f'{title}\n(Zhejiang Rank: {treated_rank}/{len(ratios_df)})', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(FIGURES_DIR / "rmspe_ratio.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # 计算p值
        treated_ratio = ratios_df[ratios_df['is_treated']]['ratio'].values[0]
        p_value = treated_rank / len(ratios_df)
        print(f"浙江RMSPE比率: {treated_ratio:.3f}")
        print(f"排名: {treated_rank}/{len(ratios_df)}")
        print(f"p值: {p_value:.4f}")
        
        return fig, ax
    
    def plot_loo(
        self,
        loo_results: pd.DataFrame,
        base_results: pd.DataFrame,
        treatment_time: int,
        title: str = 'Leave-One-Out Robustness Check',
        save_path: Optional[str] = None,
        figsize: tuple = (12, 6)
    ):
        """
        绘制留一交叉验证图
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制各排除情况的合成控制
        for excluded in loo_results['excluded_unit'].unique():
            unit_data = loo_results[loo_results['excluded_unit'] == excluded]
            ax.plot(
                unit_data['year'],
                unit_data['synthetic'],
                color=self.colors['placebo'],
                linewidth=1,
                alpha=0.5
            )
        
        # 绘制原始合成控制
        ax.plot(
            base_results['year'],
            base_results['synthetic'],
            color=self.colors['synthetic'],
            linewidth=2.5,
            linestyle='--',
            label='Original Synthetic'
        )
        
        # 绘制实际值
        ax.plot(
            base_results['year'],
            base_results['treated'],
            color=self.colors['treated'],
            linewidth=2.5,
            label='Zhejiang (Actual)'
        )
        
        # 处理时间线
        ax.axvline(
            x=treatment_time,
            color=self.colors['treatment_line'],
            linestyle=':',
            linewidth=2
        )
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Urban-Rural Income Ratio', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(FIGURES_DIR / "loo_robustness.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig, ax
    
    def create_all_plots(self, scm_model, placebo_results=None, loo_results=None):
        """生成所有图表"""
        
        print("\n生成可视化图表...")
        
        # 1. 趋势对比图
        print("  [1/5] 趋势对比图")
        self.plot_trends(
            results=scm_model.get_results(),
            treatment_time=scm_model.treatment_time,
            title='Zhejiang Common Prosperity Policy Effect\n(Synthetic Control Method)',
            outcome_label='Urban-Rural Income Ratio'
        )
        
        # 2. 处理效应图
        print("  [2/5] 处理效应图")
        self.plot_treatment_effect(
            results=scm_model.get_results(),
            treatment_time=scm_model.treatment_time
        )
        
        # 3. 权重分布图
        print("  [3/5] 权重分布图")
        self.plot_weights(weights=scm_model.get_weights())
        
        # 4. 安慰剂检验图
        if placebo_results is not None:
            print("  [4/5] 安慰剂检验图")
            self.plot_placebo(
                placebo_results=placebo_results,
                treatment_time=scm_model.treatment_time
            )
        
        # 5. 留一交叉验证图
        if loo_results is not None:
            print("  [5/5] 留一交叉验证图")
            self.plot_loo(
                loo_results=loo_results,
                base_results=scm_model.get_results(),
                treatment_time=scm_model.treatment_time
            )
        
        print(f"\n所有图表已保存至: {FIGURES_DIR}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    """可视化主函数"""
    
    print("=" * 60)
    print("合成控制法结果可视化")
    print("=" * 60)
    
    # 加载结果
    results_path = RESULTS_DIR / "scm_results.csv"
    weights_path = RESULTS_DIR / "scm_weights.csv"
    placebo_path = RESULTS_DIR / "placebo_results.csv"
    loo_path = RESULTS_DIR / "loo_results.csv"
    
    if not results_path.exists():
        print("结果文件不存在，请先运行 scm_analysis.py")
        return
    
    results = pd.read_csv(results_path)
    weights = pd.read_csv(weights_path)
    
    placebo_results = None
    if placebo_path.exists():
        placebo_results = pd.read_csv(placebo_path)
    
    loo_results = None
    if loo_path.exists():
        loo_results = pd.read_csv(loo_path)
    
    # 创建可视化器
    viz = SCMVisualizer()
    
    # 生成图表
    print("\n生成趋势对比图...")
    viz.plot_trends(
        results=results,
        treatment_time=2021,
        title='Zhejiang Common Prosperity Policy Effect\n(Synthetic Control Method)',
        outcome_label='Urban-Rural Income Ratio'
    )
    
    print("\n生成处理效应图...")
    viz.plot_treatment_effect(
        results=results,
        treatment_time=2021
    )
    
    print("\n生成权重分布图...")
    viz.plot_weights(weights=weights)
    
    if placebo_results is not None:
        print("\n生成安慰剂检验图...")
        viz.plot_placebo(
            placebo_results=placebo_results,
            treatment_time=2021
        )
        
        print("\n生成RMSPE比率分布图...")
        viz.plot_ratio_distribution(
            placebo_results=placebo_results,
            treatment_time=2021
        )
    
    if loo_results is not None:
        print("\n生成留一交叉验证图...")
        viz.plot_loo(
            loo_results=loo_results,
            base_results=results,
            treatment_time=2021
        )
    
    print("\n" + "=" * 60)
    print("可视化完成！")
    print(f"图表保存位置: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
