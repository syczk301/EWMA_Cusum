"""
CUSUM（累积和）控制图模块

本模块实现CUSUM控制图的计算和绘制功能。
CUSUM控制图对过程均值的小偏移特别敏感，能够快速检测过程变化。

作者: 质量控制系统开发团队
版本: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns


class CUSUMChart:
    """
    CUSUM（累积和）控制图类
    
    实现CUSUM控制图的计算、绘制和监控功能。
    """
    
    def __init__(self, k: float = 0.5, h: float = 5.0, target: Optional[float] = None):
        """
        初始化CUSUM控制图
        
        Args:
            k: 参考值，通常设为0.5，表示检测0.5σ的偏移
            h: 决策区间，通常设为5.0
            target: 目标值，如果为None则使用数据均值
        """
        self.k = k
        self.h = h
        self.target = target
        self.cusum_upper = None
        self.cusum_lower = None
        self.control_limits = None
        self.data = None
        
    def calculate_cusum(self, data: np.ndarray, target: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        计算CUSUM值
        
        Args:
            data: 输入数据数组
            target: 目标值，如果为None则使用数据均值
            
        Returns:
            包含上下CUSUM值的字典
        """
        if target is None:
            target = np.mean(data)
        
        self.target = target
        n = len(data)
        
        # 计算标准化偏差
        sigma = np.std(data, ddof=1)
        standardized_data = (data - target) / sigma
        
        # 计算CUSUM
        cusum_upper = np.zeros(n)
        cusum_lower = np.zeros(n)
        
        for i in range(n):
            if i == 0:
                cusum_upper[i] = max(0, standardized_data[i] - self.k)
                cusum_lower[i] = max(0, -standardized_data[i] - self.k)
            else:
                cusum_upper[i] = max(0, cusum_upper[i-1] + standardized_data[i] - self.k)
                cusum_lower[i] = max(0, cusum_lower[i-1] - standardized_data[i] - self.k)
        
        self.cusum_upper = cusum_upper
        self.cusum_lower = cusum_lower
        
        return {
            'CUSUM_upper': cusum_upper,
            'CUSUM_lower': cusum_lower
        }
    
    def calculate_control_limits(self) -> Dict[str, np.ndarray]:
        """
        计算CUSUM控制限
        
        Returns:
            控制限字典
        """
        if self.cusum_upper is None:
            raise ValueError("请先计算CUSUM值")
        
        n = len(self.cusum_upper)
        
        # CUSUM控制限
        ucl_upper = np.full(n, self.h)
        ucl_lower = np.full(n, self.h)
        cl = np.zeros(n)
        
        self.control_limits = {
            'UCL_upper': ucl_upper,
            'UCL_lower': ucl_lower,
            'CL': cl
        }
        
        return self.control_limits
    
    def fit(self, data: np.ndarray, target: Optional[float] = None) -> Dict[str, Any]:
        """
        拟合CUSUM控制图
        
        Args:
            data: 输入数据
            target: 目标值
            
        Returns:
            拟合结果字典
        """
        self.data = data
        
        # 计算CUSUM值
        cusum_values = self.calculate_cusum(data, target)
        
        # 计算控制限
        control_limits = self.calculate_control_limits()
        
        # 检测异常点
        violations = self.detect_violations(cusum_values, control_limits)
        
        # 保存违反信息到实例变量
        self.violations = violations
        
        return {
            'cusum_values': cusum_values,
            'control_limits': control_limits,
            'violations': violations,
            'target': self.target,
            'k': self.k,
            'h': self.h
        }
    
    def detect_violations(self, cusum_values: Dict[str, np.ndarray], 
                         control_limits: Dict[str, np.ndarray]) -> Dict[str, List[int]]:
        """
        检测控制限违反
        
        Args:
            cusum_values: CUSUM值字典
            control_limits: 控制限字典
            
        Returns:
            违反点索引字典
        """
        violations = {
            'upper_violations': [],
            'lower_violations': [],
            'trends': []
        }
        
        cusum_upper = cusum_values['CUSUM_upper']
        cusum_lower = cusum_values['CUSUM_lower']
        ucl_upper = control_limits['UCL_upper']
        ucl_lower = control_limits['UCL_lower']
        
        # 检测超出控制限的点
        for i in range(len(cusum_upper)):
            if cusum_upper[i] > ucl_upper[i]:
                violations['upper_violations'].append(i)
            if cusum_lower[i] > ucl_lower[i]:
                violations['lower_violations'].append(i)
        
        # 检测趋势
        violations['trends'] = self._detect_trends(cusum_values, control_limits)
        
        return violations
    
    def _detect_trends(self, cusum_values: Dict[str, np.ndarray], 
                       control_limits: Dict[str, np.ndarray]) -> List[int]:
        """
        检测趋势模式
        
        Args:
            cusum_values: CUSUM值字典
            control_limits: 控制限字典
            
        Returns:
            趋势起始点索引列表
        """
        trends = []
        cusum_upper = cusum_values['CUSUM_upper']
        cusum_lower = cusum_values['CUSUM_lower']
        
        # 检测连续上升或下降趋势
        for i in range(len(cusum_upper) - 7):
            # 检查上CUSUM连续上升
            upper_segment = cusum_upper[i:i+8]
            if np.all(np.diff(upper_segment) >= 0) and upper_segment[-1] > 0:
                trends.append(i)
            
            # 检查下CUSUM连续上升
            lower_segment = cusum_lower[i:i+8]
            if np.all(np.diff(lower_segment) >= 0) and lower_segment[-1] > 0:
                trends.append(i)
        
        return trends
    
    def plot(self, data: Optional[np.ndarray] = None, 
             show_violations: bool = True) -> 'plotly.graph_objects.Figure':
        """
        绘制CUSUM控制图
        
        Args:
            data: 原始数据，如果为None使用已拟合的数据
            show_violations: 是否显示违反点
            
        Returns:
            Plotly图形对象
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        if data is None:
            data = self.data
        
        if self.cusum_upper is None or self.control_limits is None:
            raise ValueError("请先调用fit方法")
        
        # 创建子图
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('原始数据', '上CUSUM控制图', '下CUSUM控制图'),
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        x = np.arange(len(data))
        
        # 绘制原始数据
        fig.add_trace(
            go.Scatter(
                x=x, y=data, 
                mode='lines+markers',
                name='原始数据',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4),
                showlegend=True,
                legendgroup="group1"
            ),
            row=1, col=1
        )
        
        # 添加目标值线
        fig.add_trace(
            go.Scatter(
                x=x, y=[self.target] * len(x),
                mode='lines',
                name='目标值',
                line=dict(color='#2ca02c', width=2, dash='dash'),
                showlegend=True,
                legendgroup="group2"
            ),
            row=1, col=1
        )
        
        # 绘制上CUSUM
        fig.add_trace(
            go.Scatter(
                x=x, y=self.cusum_upper,
                mode='lines+markers',
                name='上CUSUM',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=4),
                showlegend=True,
                legendgroup="group3"
            ),
            row=2, col=1
        )
        
        # 添加上CUSUM控制限
        fig.add_trace(
            go.Scatter(
                x=x, y=self.control_limits['UCL_upper'],
                mode='lines',
                name='UCL',
                line=dict(color='#d62728', width=2, dash='dash'),
                showlegend=True,
                legendgroup="group4"
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x, y=self.control_limits['CL'],
                mode='lines',
                name='中心线 (CL)',
                line=dict(color='#2ca02c', width=2, dash='dash'),
                showlegend=True,
                legendgroup="group4"
            ),
            row=2, col=1
        )
        
        # 绘制下CUSUM
        fig.add_trace(
            go.Scatter(
                x=x, y=self.cusum_lower,
                mode='lines+markers',
                name='下CUSUM',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=4),
                showlegend=True,
                legendgroup="group5"
            ),
            row=3, col=1
        )
        
        # 添加下CUSUM控制限
        fig.add_trace(
            go.Scatter(
                x=x, y=self.control_limits['UCL_lower'],
                mode='lines',
                name='UCL',
                line=dict(color='#d62728', width=2, dash='dash'),
                showlegend=False,  # 不显示在图例中，避免重复
                legendgroup="group6"
            ),
            row=3, col=1
        )
        
        # 下CUSUM的中心线不显示在图例中，避免重复
        fig.add_trace(
            go.Scatter(
                x=x, y=self.control_limits['CL'],
                mode='lines',
                name='中心线 (CL)',
                line=dict(color='#2ca02c', width=2, dash='dash'),
                showlegend=False,  # 不显示在图例中，避免重复
                legendgroup="group6"
            ),
            row=3, col=1
        )
        
        # 标记违反点
        if show_violations and hasattr(self, 'violations'):
            violations = self.violations
            if violations['upper_violations']:
                fig.add_trace(
                    go.Scatter(
                        x=violations['upper_violations'],
                        y=self.cusum_upper[violations['upper_violations']],
                        mode='markers',
                        name='上CUSUM超出UCL',
                        marker=dict(color='red', size=8, symbol='x'),
                        showlegend=True
                    ),
                    row=2, col=1
                )
            if violations['lower_violations']:
                fig.add_trace(
                    go.Scatter(
                        x=violations['lower_violations'],
                        y=self.cusum_lower[violations['lower_violations']],
                        mode='markers',
                        name='下CUSUM超出UCL',
                        marker=dict(color='red', size=8, symbol='x'),
                        showlegend=True
                    ),
                    row=3, col=1
                )
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text="CUSUM控制图",
                x=0.5,
                font=dict(size=20, color='black')
            ),
            height=900,  # 增加高度
            width=1400,  # 增加宽度
            showlegend=True,
            legend=dict(
                orientation="v",  # 改为垂直布局
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02,
                font=dict(size=12),  # 增加字体大小
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='black',
                borderwidth=1,
                itemsizing='constant'
            )
        )
        
        # 更新坐标轴
        fig.update_xaxes(title_text="样本序号", row=3, col=1)
        fig.update_yaxes(title_text="原始数据", row=1, col=1)
        fig.update_yaxes(title_text="上CUSUM值", row=2, col=1)
        fig.update_yaxes(title_text="下CUSUM值", row=3, col=1)
        
        return fig
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取CUSUM统计信息
        
        Returns:
            统计信息字典
        """
        if self.cusum_upper is None:
            return {}
        
        violations = self.violations if hasattr(self, 'violations') else {}
        
        stats = {
            'k': self.k,
            'h': self.h,
            'target': self.target,
            'cusum_upper_mean': np.mean(self.cusum_upper),
            'cusum_lower_mean': np.mean(self.cusum_lower),
            'cusum_upper_max': np.max(self.cusum_upper),
            'cusum_lower_max': np.max(self.cusum_lower),
            'total_violations': len(violations.get('upper_violations', [])) + len(violations.get('lower_violations', [])),
            'upper_violations_count': len(violations.get('upper_violations', [])),
            'lower_violations_count': len(violations.get('lower_violations', [])),
            'trend_count': len(violations.get('trends', [])),
            'process_in_control': len(violations.get('upper_violations', [])) + len(violations.get('lower_violations', [])) == 0
        }
        
        return stats
    
    def detect_shift(self, data: np.ndarray, shift_size: float = 0.5) -> Dict[str, Any]:
        """
        检测过程偏移
        
        Args:
            data: 数据数组
            shift_size: 偏移大小（以标准差为单位）
            
        Returns:
            偏移检测结果
        """
        if self.cusum_upper is None:
            raise ValueError("请先调用fit方法")
        
        # 计算ARL（平均运行长度）
        arl_in_control = self._calculate_arl_in_control()
        arl_out_of_control = self._calculate_arl_out_of_control(shift_size)
        
        # 检测偏移点
        shift_points = self._detect_shift_points(data, shift_size)
        
        return {
            'arl_in_control': arl_in_control,
            'arl_out_of_control': arl_out_of_control,
            'shift_points': shift_points,
            'detection_power': 1 / arl_out_of_control if arl_out_of_control > 0 else 0
        }
    
    def _calculate_arl_in_control(self) -> float:
        """计算受控状态下的平均运行长度"""
        # 简化计算，实际应用中可能需要更复杂的算法
        return 370.4  # 对于h=5.0的近似值
    
    def _calculate_arl_out_of_control(self, shift_size: float) -> float:
        """计算失控状态下的平均运行长度"""
        # 简化计算
        if shift_size <= 0:
            return float('inf')
        
        # 使用近似公式
        delta = shift_size
        k = self.k
        h = self.h
        
        if delta <= k:
            return float('inf')
        else:
            return np.exp(2 * (h - k) * (delta - k)) / (2 * (delta - k)**2)
    
    def _detect_shift_points(self, data: np.ndarray, shift_size: float) -> List[int]:
        """检测偏移点"""
        shift_points = []
        sigma = np.std(data, ddof=1)
        threshold = shift_size * sigma
        
        # 检测均值偏移
        for i in range(1, len(data)):
            if abs(data[i] - self.target) > threshold:
                shift_points.append(i)
        
        return shift_points
    
    def set_parameters(self, k: float = None, h: float = None, target: float = None):
        """
        设置CUSUM参数
        
        Args:
            k: 参考值
            h: 决策区间
            target: 目标值
        """
        if k is not None:
            if k <= 0:
                raise ValueError("k参数必须大于0")
            self.k = k
        
        if h is not None:
            if h <= 0:
                raise ValueError("h参数必须大于0")
            self.h = h
        
        if target is not None:
            self.target = target 