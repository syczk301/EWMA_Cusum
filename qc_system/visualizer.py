"""
可视化模块

本模块提供质量控制相关的可视化功能，包括：
- 控制图绘制
- 统计分析图表
- 交互式图表
- 报告生成

作者: 质量控制系统开发团队
版本: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import warnings
from scipy.stats import norm
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """
    可视化类
    
    提供质量控制相关的可视化功能。
    """
    
    def __init__(self):
        """初始化可视化器"""
        self.figures = {}
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
        
    def plot_data_overview(self, data: np.ndarray, title: str = "数据概览") -> go.Figure:
        """
        绘制数据概览图
        
        Args:
            data: 数据数组
            title: 图表标题
            
        Returns:
            Plotly图形对象
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('时间序列图', '直方图', '箱线图', 'Q-Q图'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 时间序列图
        fig.add_trace(
            go.Scatter(y=data, mode='lines+markers', name='数据', line=dict(color=self.colors['primary'])),
            row=1, col=1
        )
        
        # 直方图
        fig.add_trace(
            go.Histogram(x=data, nbinsx=20, name='分布', marker_color=self.colors['secondary']),
            row=1, col=2
        )
        
        # 箱线图
        fig.add_trace(
            go.Box(y=data, name='箱线图', marker_color=self.colors['success']),
            row=2, col=1
        )
        
        # Q-Q图（简化版）
        sorted_data = np.sort(data)
        theoretical_quantiles = np.percentile(np.random.normal(0, 1, 10000), np.linspace(0, 100, len(data)))
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers', 
                      name='Q-Q图', marker=dict(color=self.colors['danger'])),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            showlegend=False,
            height=700,  # 增加高度
            width=1400  # 增加宽度
        )
        
        return fig
    
    def plot_control_chart_comparison(self, data: np.ndarray, 
                                     ewma_result: Dict[str, Any],
                                     cusum_result: Dict[str, Any],
                                     title: str = "控制图对比") -> go.Figure:
        """
        绘制EWMA和CUSUM控制图对比
        
        Args:
            data: 原始数据
            ewma_result: EWMA分析结果
            cusum_result: CUSUM分析结果
            title: 图表标题
            
        Returns:
            Plotly图形对象
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('原始数据', 'EWMA控制图', 'CUSUM上控制图', 'CUSUM下控制图', '统计信息'),
            specs=[[{"colspan": 2}, None],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )
        
        # 原始数据
        fig.add_trace(
            go.Scatter(y=data, mode='lines+markers', name='原始数据', 
                      line=dict(color=self.colors['primary'])),
            row=1, col=1
        )
        
        # EWMA控制图
        ewma_values = ewma_result['ewma_values']
        ewma_limits = ewma_result['control_limits']
        x = np.arange(len(ewma_values))
        
        fig.add_trace(
            go.Scatter(x=x, y=ewma_values, mode='lines', name='EWMA值',
                      line=dict(color=self.colors['primary'])),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=ewma_limits['UCL'], mode='lines', name='UCL',
                      line=dict(color=self.colors['danger'], dash='dash')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=ewma_limits['LCL'], mode='lines', name='LCL',
                      line=dict(color=self.colors['danger'], dash='dash')),
            row=2, col=1
        )
        
        # CUSUM上控制图
        cusum_upper = cusum_result['cusum_values']['CUSUM_upper']
        cusum_limits = cusum_result['control_limits']
        
        fig.add_trace(
            go.Scatter(x=x, y=cusum_upper, mode='lines', name='上CUSUM',
                      line=dict(color=self.colors['secondary'])),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=x, y=cusum_limits['UCL_upper'], mode='lines', name='UCL',
                      line=dict(color=self.colors['danger'], dash='dash')),
            row=2, col=2
        )
        
        # CUSUM下控制图
        cusum_lower = cusum_result['cusum_values']['CUSUM_lower']
        
        fig.add_trace(
            go.Scatter(x=x, y=cusum_lower, mode='lines', name='下CUSUM',
                      line=dict(color=self.colors['warning'])),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=cusum_limits['UCL_lower'], mode='lines', name='UCL',
                      line=dict(color=self.colors['danger'], dash='dash')),
            row=3, col=1
        )
        
        # 统计信息表格
        ewma_stats = ewma_result.get('statistics', {})
        cusum_stats = cusum_result.get('statistics', {})
        
        stats_table = go.Table(
            header=dict(values=['指标', 'EWMA', 'CUSUM'],
                       fill_color=self.colors['info'],
                       font=dict(color='white', size=12)),
            cells=dict(values=[
                ['违反次数', '趋势次数', '过程受控'],
                [ewma_stats.get('total_violations', 0), 
                 ewma_stats.get('trend_count', 0),
                 '是' if ewma_stats.get('process_in_control', True) else '否'],
                [cusum_stats.get('total_violations', 0),
                 cusum_stats.get('trend_count', 0),
                 '是' if cusum_stats.get('process_in_control', True) else '否']
            ])
        )
        
        fig.add_trace(stats_table, row=3, col=2)
        
        fig.update_layout(
            title=title,
            height=900,  # 增加高度
            width=1400,  # 增加宽度
            showlegend=True
        )
        
        return fig
    
    def plot_capability_analysis(self, data: np.ndarray, 
                                capability_result: Dict[str, Any],
                                title: str = "过程能力分析") -> go.Figure:
        """
        绘制过程能力分析图
        
        Args:
            data: 数据数组
            capability_result: 过程能力分析结果
            title: 图表标题
            
        Returns:
            Plotly图形对象
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('数据分布', '过程能力指数', '规格限分析', '不合格品率'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "indicator"}, {"secondary_y": False}]]
        )
        
        # 数据分布直方图
        fig.add_trace(
            go.Histogram(x=data, nbinsx=20, name='数据分布',
                        marker_color=self.colors['primary']),
            row=1, col=1
        )
        
        # 添加规格限线
        specs = capability_result['specifications']
        if specs['usl'] is not None:
            fig.add_vline(x=specs['usl'], line_dash="dash", line_color="red",
                         annotation_text="USL", row=1, col=1)
        if specs['lsl'] is not None:
            fig.add_vline(x=specs['lsl'], line_dash="dash", line_color="red",
                         annotation_text="LSL", row=1, col=1)
        if specs['target'] is not None:
            fig.add_vline(x=specs['target'], line_dash="dash", line_color="green",
                         annotation_text="Target", row=1, col=1)
        
        # 过程能力指数条形图
        capability_indices = capability_result['capability_indices']
        if capability_indices:
            indices = list(capability_indices.keys())
            values = list(capability_indices.values())
            
            fig.add_trace(
                go.Bar(x=indices, y=values, name='能力指数',
                      marker_color=self.colors['secondary']),
                row=1, col=2
            )
            
            # 添加参考线
            fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                         annotation_text="Cp=1.0", row=1, col=2)
            fig.add_hline(y=1.33, line_dash="dash", line_color="green",
                         annotation_text="Cp=1.33", row=1, col=2)
        
        # 规格限分析
        basic_stats = capability_result['basic_stats']
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=basic_stats['mean'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "均值"},
                delta={'reference': specs.get('target', basic_stats['mean'])},
                gauge={'axis': {'range': [None, basic_stats['max']]},
                       'bar': {'color': self.colors['primary']},
                       'steps': [{'range': [0, basic_stats['min']], 'color': "lightgray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': basic_stats['max']}}
            ),
            row=2, col=1
        )
        
        # 不合格品率
        defect_rate = capability_result['defect_rate']
        if defect_rate:
            defect_types = list(defect_rate.keys())
            defect_values = list(defect_rate.values())
            
            fig.add_trace(
                go.Bar(x=defect_types, y=defect_values, name='不合格品率',
                      marker_color=self.colors['danger']),
                row=2, col=2
            )
        
        fig.update_layout(
            title=title,
            height=600
        )
        
        return fig
    
    def plot_trend_analysis(self, data: np.ndarray, 
                           trend_result: Dict[str, Any],
                           title: str = "趋势分析") -> go.Figure:
        """
        绘制趋势分析图
        
        Args:
            data: 数据数组
            trend_result: 趋势分析结果
            title: 图表标题
            
        Returns:
            Plotly图形对象
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('时间序列', '移动平均', '线性趋势', '趋势变化点'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        x = np.arange(len(data))
        
        # 时间序列
        fig.add_trace(
            go.Scatter(x=x, y=data, mode='lines+markers', name='原始数据',
                      line=dict(color=self.colors['primary'])),
            row=1, col=1
        )
        
        # 移动平均
        if 'moving_average' in trend_result:
            moving_avg = trend_result['moving_average']
            x_ma = np.arange(len(moving_avg))
            fig.add_trace(
                go.Scatter(x=x_ma, y=moving_avg, mode='lines', name='移动平均',
                          line=dict(color=self.colors['secondary'])),
                row=1, col=2
            )
        
        # 线性趋势
        slope = trend_result['slope']
        intercept = trend_result['intercept']
        trend_line = slope * x + intercept
        
        fig.add_trace(
            go.Scatter(x=x, y=data, mode='markers', name='数据点',
                      marker=dict(color=self.colors['primary'])),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=trend_line, mode='lines', name='趋势线',
                      line=dict(color=self.colors['danger'])),
            row=2, col=1
        )
        
        # 趋势变化点
        change_points = trend_result.get('change_points', [])
        if change_points:
            fig.add_trace(
                go.Scatter(x=x, y=data, mode='lines+markers', name='数据',
                          line=dict(color=self.colors['primary'])),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=change_points, y=data[change_points], mode='markers',
                          name='变化点', marker=dict(color=self.colors['danger'], size=10)),
                row=2, col=2
            )
        
        # 添加趋势信息
        trend_info = f"趋势方向: {trend_result['trend_direction']}<br>"
        trend_info += f"趋势强度: {trend_result['trend_strength']:.3f}<br>"
        trend_info += f"显著性: {'是' if trend_result['trend_significant'] else '否'}"
        
        fig.add_annotation(
            text=trend_info,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title=title,
            height=700,  # 增加高度
            width=1400  # 增加宽度
        )
        
        return fig
    
    def plot_normality_test(self, data: np.ndarray, 
                           normality_result: Dict[str, Any],
                           title: str = "正态性检验") -> go.Figure:
        """
        绘制正态性检验图
        
        Args:
            data: 数据数组
            normality_result: 正态性检验结果
            title: 图表标题
            
        Returns:
            Plotly图形对象
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Q-Q图', 'P-P图', '检验结果表格'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "table"}, {"secondary_y": False}]]
        )
        
        # 直方图
        fig.add_trace(
            go.Histogram(x=data, nbinsx=20, name='数据分布',
                        marker_color=self.colors['primary']),
            row=1, col=1
        )
        
        # Q-Q图
        sorted_data = np.sort(data)
        theoretical_quantiles = np.percentile(np.random.normal(0, 1, 10000), 
                                            np.linspace(0, 100, len(data)))
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers',
                      name='Q-Q图', marker=dict(color=self.colors['danger'])),
            row=1, col=1
        )
        
        # 添加对角线
        min_val = min(theoretical_quantiles.min(), sorted_data.min())
        max_val = max(theoretical_quantiles.max(), sorted_data.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                      name='对角线', line=dict(color=self.colors['danger'], dash='dash')),
            row=1, col=1
        )
        
        # P-P图
        empirical_cdf = np.arange(1, len(data)+1) / len(data)
        theoretical_cdf = norm.cdf(sorted_data, np.mean(data), np.std(data))
        
        fig.add_trace(
            go.Scatter(x=theoretical_cdf, y=empirical_cdf, mode='markers',
                      name='P-P图', marker=dict(color=self.colors['warning'])),
            row=1, col=2
        )
        
        # 添加对角线
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                      name='对角线', line=dict(color=self.colors['danger'], dash='dash')),
            row=1, col=2
        )
        
        # 检验结果表格
        shapiro = normality_result['shapiro_wilk']
        ks = normality_result['kolmogorov_smirnov']
        
        test_table = go.Table(
            header=dict(values=['检验方法', '统计量', 'P值', '结论'],
                       fill_color=self.colors['info'],
                       font=dict(color='white', size=12)),
            cells=dict(values=[
                ['Shapiro-Wilk', 'Kolmogorov-Smirnov'],
                [f"{shapiro['statistic']:.4f}", f"{ks['statistic']:.4f}"],
                [f"{shapiro['p_value']:.4f}", f"{ks['p_value']:.4f}"],
                ['正态' if shapiro['p_value'] > 0.05 else '非正态',
                 '正态' if ks['p_value'] > 0.05 else '非正态']
            ])
        )
        
        fig.add_trace(test_table, row=2, col=1)
        
        fig.update_layout(
            title=title,
            height=700,  # 增加高度
            width=1400  # 增加宽度
        )
        
        return fig
    
    def create_dashboard(self, data: np.ndarray, 
                        ewma_result: Dict[str, Any],
                        cusum_result: Dict[str, Any],
                        capability_result: Dict[str, Any],
                        trend_result: Dict[str, Any]) -> go.Figure:
        """
        创建综合仪表板
        
        Args:
            data: 原始数据
            ewma_result: EWMA分析结果
            cusum_result: CUSUM分析结果
            capability_result: 过程能力分析结果
            trend_result: 趋势分析结果
            
        Returns:
            Plotly仪表板图形
        """
        # 创建子图
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('数据概览', 'EWMA控制图', 'CUSUM控制图',
                           '过程能力', '趋势分析', '正态性检验',
                           '统计摘要', '异常检测', '控制限违反'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "table"}, {"type": "scatter"}, {"type": "table"}]]
        )
        
        # 数据概览
        fig.add_trace(
            go.Scatter(y=data, mode='lines+markers', name='数据',
                      line=dict(color=self.colors['primary'])),
            row=1, col=1
        )
        
        # EWMA控制图
        if 'ewma_values' in ewma_result:
            ewma_values = ewma_result['ewma_values']
            ewma_limits = ewma_result['control_limits']
            x = np.arange(len(ewma_values))
            
            fig.add_trace(
                go.Scatter(x=x, y=ewma_values, mode='lines', name='EWMA',
                          line=dict(color=self.colors['primary'])),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=x, y=ewma_limits['UCL'], mode='lines', name='UCL',
                          line=dict(color=self.colors['danger'], dash='dash')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=x, y=ewma_limits['LCL'], mode='lines', name='LCL',
                          line=dict(color=self.colors['danger'], dash='dash')),
                row=1, col=2
            )
        
        # CUSUM控制图
        if 'cusum_values' in cusum_result:
            cusum_upper = cusum_result['cusum_values']['CUSUM_upper']
            cusum_lower = cusum_result['cusum_values']['CUSUM_lower']
            x = np.arange(len(cusum_upper))
            
            fig.add_trace(
                go.Scatter(x=x, y=cusum_upper, mode='lines', name='上CUSUM',
                          line=dict(color=self.colors['secondary'])),
                row=1, col=3
            )
            fig.add_trace(
                go.Scatter(x=x, y=cusum_lower, mode='lines', name='下CUSUM',
                          line=dict(color=self.colors['warning'])),
                row=1, col=3
            )
        
        # 过程能力
        if 'capability_indices' in capability_result:
            indices = list(capability_result['capability_indices'].keys())
            values = list(capability_result['capability_indices'].values())
            
            fig.add_trace(
                go.Bar(x=indices, y=values, name='能力指数',
                      marker_color=self.colors['success']),
                row=2, col=1
            )
        
        # 趋势分析
        if 'moving_average' in trend_result:
            moving_avg = trend_result['moving_average']
            x_ma = np.arange(len(moving_avg))
            
            fig.add_trace(
                go.Scatter(x=x_ma, y=moving_avg, mode='lines', name='移动平均',
                          line=dict(color=self.colors['secondary'])),
                row=2, col=2
            )
        
        # 正态性检验
        sorted_data = np.sort(data)
        theoretical_quantiles = np.percentile(np.random.normal(0, 1, 10000), 
                                            np.linspace(0, 100, len(data)))
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers',
                      name='Q-Q图', marker=dict(color=self.colors['info'])),
            row=2, col=3
        )
        
        # 统计摘要表格
        basic_stats = capability_result.get('basic_stats', {})
        summary_table = go.Table(
            header=dict(values=['统计量', '值'],
                       fill_color=self.colors['info'],
                       font=dict(color='white', size=10)),
            cells=dict(values=[
                ['样本数', '均值', '标准差', '最小值', '最大值'],
                [len(data), f"{basic_stats.get('mean', 0):.2f}", 
                 f"{basic_stats.get('std', 0):.2f}",
                 f"{basic_stats.get('min', 0):.2f}",
                 f"{basic_stats.get('max', 0):.2f}"]
            ])
        )
        
        fig.add_trace(summary_table, row=3, col=1)
        
        # 异常检测
        fig.add_trace(
            go.Histogram(x=data, nbinsx=15, name='分布',
                        marker_color=self.colors['primary']),
            row=3, col=2
        )
        
        # 控制限违反
        ewma_violations = ewma_result.get('violations', {})
        cusum_violations = cusum_result.get('violations', {})
        
        violation_data = [
            ['EWMA超出UCL', 'EWMA低于LCL', 'CUSUM上违反', 'CUSUM下违反'],
            [len(ewma_violations.get('above_ucl', [])),
             len(ewma_violations.get('below_lcl', [])),
             len(cusum_violations.get('upper_violations', [])),
             len(cusum_violations.get('lower_violations', []))]
        ]
        
        violation_table = go.Table(
            header=dict(values=['违反类型', '次数'],
                       fill_color=self.colors['danger'],
                       font=dict(color='white', size=10)),
            cells=dict(values=violation_data)
        )
        
        fig.add_trace(violation_table, row=3, col=3)
        
        fig.update_layout(
            title="质量控制综合仪表板",
            height=1000,  # 增加高度
            width=1600,  # 增加宽度
            showlegend=True
        )
        
        return fig 