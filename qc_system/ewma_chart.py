"""
EWMA（指数加权移动平均）控制图模块

本模块实现EWMA控制图的计算和绘制功能。
EWMA控制图对过程均值的小偏移特别敏感。

作者: 质量控制系统开发团队
版本: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns


class EWMAChart:
    """
    EWMA（指数加权移动平均）控制图类
    
    实现EWMA控制图的计算、绘制和监控功能。
    """
    
    def __init__(self, lambda_param: float = 0.2, k: float = 3.0):
        """
        初始化EWMA控制图
        
        Args:
            lambda_param: 平滑参数λ，取值范围[0,1]，默认0.2
            k: 控制限系数，默认3.0
        """
        self.lambda_param = lambda_param
        self.k = k
        self.ewma_values = None
        self.control_limits = None
        self.center_line = None
        self.data = None
        
    def calculate_ewma(self, data: np.ndarray, target: Optional[float] = None) -> np.ndarray:
        """
        计算EWMA值
        
        Args:
            data: 输入数据数组
            target: 目标值，如果为None则使用数据均值
            
        Returns:
            EWMA值数组
        """
        if target is None:
            target = np.mean(data)
        
        n = len(data)
        ewma = np.zeros(n)
        ewma[0] = target
        
        for i in range(1, n):
            ewma[i] = self.lambda_param * data[i-1] + (1 - self.lambda_param) * ewma[i-1]
        
        self.ewma_values = ewma
        self.center_line = target
        return ewma
    
    def calculate_control_limits(self, data: np.ndarray, target: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        计算EWMA控制限
        
        Args:
            data: 输入数据数组
            target: 目标值
            
        Returns:
            包含控制限的字典
        """
        if target is None:
            target = np.mean(data)
        
        sigma = np.std(data, ddof=1)
        n = len(data)
        
        # 计算控制限
        ucl = np.zeros(n)
        lcl = np.zeros(n)
        cl = np.full(n, target)
        
        for i in range(n):
            # 计算累积方差
            var_factor = self.lambda_param * (1 - (1 - self.lambda_param)**(2*(i+1))) / (2 - self.lambda_param)
            sigma_ewma = sigma * np.sqrt(var_factor)
            
            ucl[i] = target + self.k * sigma_ewma
            lcl[i] = target - self.k * sigma_ewma
        
        self.control_limits = {
            'UCL': ucl,
            'LCL': lcl,
            'CL': cl
        }
        
        return self.control_limits
    
    def fit(self, data: np.ndarray, target: Optional[float] = None) -> Dict[str, Any]:
        """
        拟合EWMA控制图
        
        Args:
            data: 输入数据
            target: 目标值
            
        Returns:
            拟合结果字典
        """
        self.data = data
        
        # 计算EWMA值
        ewma_values = self.calculate_ewma(data, target)
        
        # 计算控制限
        control_limits = self.calculate_control_limits(data, target)
        
        # 检测异常点
        violations = self.detect_violations(ewma_values, control_limits)
        
        # 保存违反信息到实例变量
        self.violations = violations
        
        return {
            'ewma_values': ewma_values,
            'control_limits': control_limits,
            'violations': violations,
            'target': self.center_line,
            'lambda_param': self.lambda_param,
            'k': self.k
        }
    
    def detect_violations(self, ewma_values: np.ndarray, control_limits: Dict[str, np.ndarray]) -> Dict[str, List[int]]:
        """
        检测控制限违反
        
        Args:
            ewma_values: EWMA值
            control_limits: 控制限字典
            
        Returns:
            违反点索引字典
        """
        violations = {
            'above_ucl': [],
            'below_lcl': [],
            'trends': []
        }
        
        ucl = control_limits['UCL']
        lcl = control_limits['LCL']
        
        # 检测超出控制限的点
        for i in range(len(ewma_values)):
            if ewma_values[i] > ucl[i]:
                violations['above_ucl'].append(i)
            elif ewma_values[i] < lcl[i]:
                violations['below_lcl'].append(i)
        
        # 检测趋势（连续8个点在同一侧）
        violations['trends'] = self._detect_trends(ewma_values, control_limits)
        
        return violations
    
    def _detect_trends(self, ewma_values: np.ndarray, control_limits: Dict[str, np.ndarray]) -> List[int]:
        """
        检测趋势模式
        
        Args:
            ewma_values: EWMA值
            control_limits: 控制限
            
        Returns:
            趋势起始点索引列表
        """
        trends = []
        cl = control_limits['CL']
        
        # 检测连续8个点在中心线同一侧
        for i in range(len(ewma_values) - 7):
            segment = ewma_values[i:i+8]
            center_segment = cl[i:i+8]
            
            # 检查是否都在中心线之上或之下
            if np.all(segment > center_segment) or np.all(segment < center_segment):
                trends.append(i)
        
        return trends
    
    def plot(self, data: Optional[np.ndarray] = None, 
             show_violations: bool = True) -> 'plotly.graph_objects.Figure':
        """
        绘制EWMA控制图
        
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
        
        if self.ewma_values is None or self.control_limits is None:
            raise ValueError("请先调用fit方法")
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('原始数据', 'EWMA控制图'),
            vertical_spacing=0.1,
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
                x=x, y=[self.center_line] * len(x),
                mode='lines',
                name='目标值',
                line=dict(color='#2ca02c', width=2, dash='dash'),
                showlegend=True,
                legendgroup="group2"
            ),
            row=1, col=1
        )
        
        # 绘制EWMA控制图
        fig.add_trace(
            go.Scatter(
                x=x, y=self.ewma_values,
                mode='lines+markers',
                name='EWMA值',
                line=dict(color='#ff7f0e', width=3),  # 改为橙色以区分
                marker=dict(size=4),
                showlegend=True,
                legendgroup="group3"
            ),
            row=2, col=1
        )
        
        # 添加控制限
        fig.add_trace(
            go.Scatter(
                x=x, y=self.control_limits['UCL'],
                mode='lines',
                name='上控制限 (UCL)',
                line=dict(color='#d62728', width=2, dash='dash'),
                showlegend=True,
                legendgroup="group4"
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x, y=self.control_limits['LCL'],
                mode='lines',
                name='下控制限 (LCL)',
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
        
        # 标记违反点
        if show_violations and hasattr(self, 'violations'):
            violations = self.violations
            if violations['above_ucl']:
                fig.add_trace(
                    go.Scatter(
                        x=violations['above_ucl'],
                        y=self.ewma_values[violations['above_ucl']],
                        mode='markers',
                        name='超出UCL',
                        marker=dict(color='red', size=8, symbol='x'),
                        showlegend=True
                    ),
                    row=2, col=1
                )
            if violations['below_lcl']:
                fig.add_trace(
                    go.Scatter(
                        x=violations['below_lcl'],
                        y=self.ewma_values[violations['below_lcl']],
                        mode='markers',
                        name='低于LCL',
                        marker=dict(color='red', size=8, symbol='x'),
                        showlegend=True
                    ),
                    row=2, col=1
                )
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text="EWMA控制图",
                x=0.5,
                font=dict(size=20, color='black')
            ),
            height=700,  # 增加高度
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
        fig.update_xaxes(title_text="样本序号", row=2, col=1)
        fig.update_yaxes(title_text="原始数据", row=1, col=1)
        fig.update_yaxes(title_text="EWMA值", row=2, col=1)
        
        return fig
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取EWMA统计信息
        
        Returns:
            统计信息字典
        """
        if self.ewma_values is None:
            return {}
        
        violations = self.violations if hasattr(self, 'violations') else {}
        
        stats = {
            'lambda_param': self.lambda_param,
            'k': self.k,
            'target': self.center_line,
            'ewma_mean': np.mean(self.ewma_values),
            'ewma_std': np.std(self.ewma_values),
            'total_violations': len(violations.get('above_ucl', [])) + len(violations.get('below_lcl', [])),
            'above_ucl_count': len(violations.get('above_ucl', [])),
            'below_lcl_count': len(violations.get('below_lcl', [])),
            'trend_count': len(violations.get('trends', [])),
            'process_in_control': len(violations.get('above_ucl', [])) + len(violations.get('below_lcl', [])) == 0
        }
        
        return stats
    
    def predict_next_ewma(self, next_value: float) -> float:
        """
        预测下一个EWMA值
        
        Args:
            next_value: 下一个观测值
            
        Returns:
            预测的EWMA值
        """
        if self.ewma_values is None:
            raise ValueError("请先调用fit方法")
        
        last_ewma = self.ewma_values[-1]
        next_ewma = self.lambda_param * next_value + (1 - self.lambda_param) * last_ewma
        
        return next_ewma
    
    def set_parameters(self, lambda_param: float = None, k: float = None):
        """
        设置EWMA参数
        
        Args:
            lambda_param: 平滑参数λ
            k: 控制限系数
        """
        if lambda_param is not None:
            if not 0 <= lambda_param <= 1:
                raise ValueError("λ参数必须在[0,1]范围内")
            self.lambda_param = lambda_param
        
        if k is not None:
            if k <= 0:
                raise ValueError("k参数必须大于0")
            self.k = k 