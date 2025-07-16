"""
统计分析模块

本模块提供质量控制相关的统计分析功能，包括：
- 过程能力分析
- 趋势分析
- 异常检测
- 统计报告生成

作者: 质量控制系统开发团队
版本: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import norm, chi2
import warnings
warnings.filterwarnings('ignore')


class Statistics:
    """
    统计分析类
    
    提供质量控制相关的统计分析功能。
    """
    
    def __init__(self):
        """初始化统计分析器"""
        self.data = None
        self.results = {}
        
    def process_capability_analysis(self, data: np.ndarray, 
                                  usl: Optional[float] = None,
                                  lsl: Optional[float] = None,
                                  target: Optional[float] = None) -> Dict[str, Any]:
        """
        过程能力分析
        
        Args:
            data: 数据数组
            usl: 上规格限
            lsl: 下规格限
            target: 目标值
            
        Returns:
            过程能力分析结果
        """
        if target is None:
            target = np.mean(data)
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        # 基本统计量
        basic_stats = {
            'mean': mean,
            'std': std,
            'median': np.median(data),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.max(data) - np.min(data),
            'cv': std / mean if mean != 0 else 0,  # 变异系数
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }
        
        # 过程能力指数
        capability_indices = {}
        
        if usl is not None and lsl is not None:
            # 双边规格限
            tolerance = usl - lsl
            capability_indices['Cp'] = tolerance / (6 * std) if std > 0 else 0
            capability_indices['Cpu'] = (usl - mean) / (3 * std) if std > 0 else 0
            capability_indices['Cpl'] = (mean - lsl) / (3 * std) if std > 0 else 0
            capability_indices['Cpk'] = min(capability_indices['Cpu'], capability_indices['Cpl'])
            
        elif usl is not None:
            # 只有上规格限
            capability_indices['Cpu'] = (usl - mean) / (3 * std) if std > 0 else 0
            capability_indices['Cpk'] = capability_indices['Cpu']
            
        elif lsl is not None:
            # 只有下规格限
            capability_indices['Cpl'] = (mean - lsl) / (3 * std) if std > 0 else 0
            capability_indices['Cpk'] = capability_indices['Cpl']
        
        # 过程性能指数
        if usl is not None and lsl is not None:
            tolerance = usl - lsl
            capability_indices['Pp'] = tolerance / (6 * std) if std > 0 else 0
            capability_indices['Ppu'] = (usl - mean) / (3 * std) if std > 0 else 0
            capability_indices['Ppl'] = (mean - lsl) / (3 * std) if std > 0 else 0
            capability_indices['Ppk'] = min(capability_indices['Ppu'], capability_indices['Ppl'])
        
        # 不合格品率估计
        defect_rate = self._estimate_defect_rate(data, usl, lsl, mean, std)
        
        return {
            'basic_stats': basic_stats,
            'capability_indices': capability_indices,
            'defect_rate': defect_rate,
            'specifications': {
                'usl': usl,
                'lsl': lsl,
                'target': target
            }
        }
    
    def _estimate_defect_rate(self, data: np.ndarray, usl: Optional[float], 
                             lsl: Optional[float], mean: float, std: float) -> Dict[str, float]:
        """估计不合格品率"""
        defect_rate = {}
        
        if usl is not None:
            # 超出上规格限的比例
            above_usl = np.sum(data > usl) / len(data)
            defect_rate['above_usl'] = above_usl
            
            # 理论超出上规格限的比例
            if std > 0:
                z_usl = (usl - mean) / std
                theoretical_above_usl = 1 - norm.cdf(z_usl)
                defect_rate['theoretical_above_usl'] = theoretical_above_usl
        
        if lsl is not None:
            # 低于下规格限的比例
            below_lsl = np.sum(data < lsl) / len(data)
            defect_rate['below_lsl'] = below_lsl
            
            # 理论低于下规格限的比例
            if std > 0:
                z_lsl = (lsl - mean) / std
                theoretical_below_lsl = norm.cdf(z_lsl)
                defect_rate['theoretical_below_lsl'] = theoretical_below_lsl
        
        if usl is not None and lsl is not None:
            # 总不合格品率
            total_defect = np.sum((data < lsl) | (data > usl)) / len(data)
            defect_rate['total_defect'] = total_defect
            
            # 理论总不合格品率
            if std > 0:
                theoretical_total = defect_rate.get('theoretical_below_lsl', 0) + defect_rate.get('theoretical_above_usl', 0)
                defect_rate['theoretical_total'] = theoretical_total
        
        return defect_rate
    
    def trend_analysis(self, data: np.ndarray, window_size: int = 5) -> Dict[str, Any]:
        """
        趋势分析
        
        Args:
            data: 数据数组
            window_size: 移动窗口大小
            
        Returns:
            趋势分析结果
        """
        n = len(data)
        
        # 移动平均
        if window_size > n:
            window_size = n
        
        moving_avg = []
        for i in range(n - window_size + 1):
            moving_avg.append(np.mean(data[i:i+window_size]))
        
        # 线性趋势
        x = np.arange(n)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        # 趋势强度
        trend_strength = abs(r_value)
        
        # 趋势方向
        if slope > 0:
            trend_direction = "上升"
        elif slope < 0:
            trend_direction = "下降"
        else:
            trend_direction = "无趋势"
        
        # 趋势显著性
        if p_value < 0.05:
            trend_significant = True
        else:
            trend_significant = False
        
        # 检测趋势变化点
        change_points = self._detect_trend_changes(data, window_size)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'trend_significant': trend_significant,
            'moving_average': moving_avg,
            'change_points': change_points
        }
    
    def _detect_trend_changes(self, data: np.ndarray, window_size: int) -> List[int]:
        """检测趋势变化点"""
        change_points = []
        n = len(data)
        
        if n < 2 * window_size:
            return change_points
        
        # 使用移动窗口检测趋势变化
        for i in range(window_size, n - window_size):
            # 前半段趋势
            first_half = data[i-window_size:i]
            second_half = data[i:i+window_size]
            
            # 计算两段的斜率
            x1 = np.arange(len(first_half))
            x2 = np.arange(len(second_half))
            
            slope1, _, _, p1, _ = stats.linregress(x1, first_half)
            slope2, _, _, p2, _ = stats.linregress(x2, second_half)
            
            # 如果斜率变化显著，认为是变化点
            if abs(slope2 - slope1) > 0.1 and p1 < 0.05 and p2 < 0.05:
                change_points.append(i)
        
        return change_points
    
    def normality_test(self, data: np.ndarray) -> Dict[str, Any]:
        """
        正态性检验
        
        Args:
            data: 数据数组
            
        Returns:
            正态性检验结果
        """
        # Shapiro-Wilk检验
        shapiro_stat, shapiro_p = stats.shapiro(data)
        
        # Kolmogorov-Smirnov检验
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        
        # Anderson-Darling检验
        anderson_result = stats.anderson(data)
        
        # 判断是否服从正态分布
        is_normal = shapiro_p > 0.05 and ks_p > 0.05
        
        return {
            'shapiro_wilk': {
                'statistic': shapiro_stat,
                'p_value': shapiro_p
            },
            'kolmogorov_smirnov': {
                'statistic': ks_stat,
                'p_value': ks_p
            },
            'anderson_darling': {
                'statistic': anderson_result.statistic,
                'critical_values': anderson_result.critical_values,
                'significance_level': anderson_result.significance_level
            },
            'is_normal': is_normal
        }
    
    def autocorrelation_analysis(self, data: np.ndarray, max_lag: int = 10) -> Dict[str, Any]:
        """
        自相关分析
        
        Args:
            data: 数据数组
            max_lag: 最大滞后阶数
            
        Returns:
            自相关分析结果
        """
        n = len(data)
        if max_lag >= n:
            max_lag = n - 1
        
        # 计算自相关系数
        autocorr = []
        for lag in range(1, max_lag + 1):
            # 计算滞后自相关系数
            corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            autocorr.append(corr)
        
        # 计算偏自相关系数
        pacf = self._calculate_pacf(data, max_lag)
        
        # 检测显著的自相关
        significant_lags = []
        for i, ac in enumerate(autocorr):
            if abs(ac) > 2 / np.sqrt(n):  # 95%置信区间
                significant_lags.append(i + 1)
        
        return {
            'autocorrelation': autocorr,
            'partial_autocorrelation': pacf,
            'significant_lags': significant_lags,
            'has_autocorrelation': len(significant_lags) > 0
        }
    
    def _calculate_pacf(self, data: np.ndarray, max_lag: int) -> List[float]:
        """计算偏自相关系数"""
        pacf = []
        n = len(data)
        
        for lag in range(1, max_lag + 1):
            if lag == 1:
                # 一阶偏自相关系数等于一阶自相关系数
                pacf.append(np.corrcoef(data[:-1], data[1:])[0, 1])
            else:
                # 使用Yule-Walker方程计算高阶偏自相关系数
                # 这里使用简化方法
                pacf.append(0.0)  # 简化实现
        
        return pacf
    
    def outlier_detection(self, data: np.ndarray, method: str = 'iqr') -> Dict[str, Any]:
        """
        异常值检测
        
        Args:
            data: 数据数组
            method: 检测方法 ('iqr', 'zscore', 'isolation_forest')
            
        Returns:
            异常值检测结果
        """
        outliers = {}
        
        if method == 'iqr':
            outliers = self._detect_outliers_iqr(data)
        elif method == 'zscore':
            outliers = self._detect_outliers_zscore(data)
        elif method == 'isolation_forest':
            outliers = self._detect_outliers_isolation_forest(data)
        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")
        
        return outliers
    
    def _detect_outliers_iqr(self, data: np.ndarray) -> Dict[str, Any]:
        """使用IQR方法检测异常值"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
        
        return {
            'outlier_indices': outlier_indices.tolist(),
            'outlier_values': data[outlier_indices].tolist(),
            'outlier_count': len(outlier_indices),
            'outlier_percentage': len(outlier_indices) / len(data) * 100,
            'bounds': {
                'lower': lower_bound,
                'upper': upper_bound
            }
        }
    
    def _detect_outliers_zscore(self, data: np.ndarray, threshold: float = 3.0) -> Dict[str, Any]:
        """使用Z-score方法检测异常值"""
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        outlier_indices = np.where(z_scores > threshold)[0]
        
        return {
            'outlier_indices': outlier_indices.tolist(),
            'outlier_values': data[outlier_indices].tolist(),
            'outlier_count': len(outlier_indices),
            'outlier_percentage': len(outlier_indices) / len(data) * 100,
            'z_scores': z_scores.tolist(),
            'threshold': threshold
        }
    
    def _detect_outliers_isolation_forest(self, data: np.ndarray) -> Dict[str, Any]:
        """使用隔离森林方法检测异常值"""
        try:
            from sklearn.ensemble import IsolationForest
            
            # 重塑数据为2D数组
            data_2d = data.reshape(-1, 1)
            
            # 训练隔离森林模型
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(data_2d)
            
            # 异常值索引
            outlier_indices = np.where(predictions == -1)[0]
            
            return {
                'outlier_indices': outlier_indices.tolist(),
                'outlier_values': data[outlier_indices].tolist(),
                'outlier_count': len(outlier_indices),
                'outlier_percentage': len(outlier_indices) / len(data) * 100
            }
            
        except ImportError:
            print("警告: sklearn未安装，使用IQR方法替代")
            return self._detect_outliers_iqr(data)
    
    def generate_report(self, data: np.ndarray, 
                       usl: Optional[float] = None,
                       lsl: Optional[float] = None,
                       target: Optional[float] = None) -> Dict[str, Any]:
        """
        生成综合统计报告
        
        Args:
            data: 数据数组
            usl: 上规格限
            lsl: 下规格限
            target: 目标值
            
        Returns:
            综合统计报告
        """
        report = {}
        
        # 过程能力分析
        report['capability_analysis'] = self.process_capability_analysis(data, usl, lsl, target)
        
        # 趋势分析
        report['trend_analysis'] = self.trend_analysis(data)
        
        # 正态性检验
        report['normality_test'] = self.normality_test(data)
        
        # 自相关分析
        report['autocorrelation_analysis'] = self.autocorrelation_analysis(data)
        
        # 异常值检测
        report['outlier_detection'] = self.outlier_detection(data)
        
        # 数据摘要
        report['data_summary'] = {
            'sample_size': len(data),
            'mean': np.mean(data),
            'std': np.std(data, ddof=1),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.max(data) - np.min(data),
            'median': np.median(data),
            'q1': np.percentile(data, 25),
            'q3': np.percentile(data, 75)
        }
        
        return report 