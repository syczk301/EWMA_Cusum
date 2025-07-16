"""
辅助工具函数

本模块包含质量控制系统的辅助工具函数。

作者: 质量控制系统开发团队
版本: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import json
import datetime
from pathlib import Path


def generate_sample_data(n_samples: int = 100, 
                        mean: float = 100.0, 
                        std: float = 5.0,
                        trend: float = 0.0,
                        outliers: bool = False) -> np.ndarray:
    """
    生成示例数据
    
    Args:
        n_samples: 样本数量
        mean: 均值
        std: 标准差
        trend: 趋势系数
        outliers: 是否添加异常值
        
    Returns:
        生成的示例数据
    """
    # 生成基础数据
    base_data = np.random.normal(mean, std, n_samples)
    
    # 添加趋势
    if trend != 0:
        trend_component = trend * np.arange(n_samples)
        base_data += trend_component
    
    # 添加异常值
    if outliers:
        outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        base_data[outlier_indices] += np.random.normal(0, std * 3, len(outlier_indices))
    
    return base_data


def calculate_control_limits(data: np.ndarray, method: str = '3sigma') -> Dict[str, float]:
    """
    计算控制限
    
    Args:
        data: 数据数组
        method: 计算方法 ('3sigma', 'percentile')
        
    Returns:
        控制限字典
    """
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    if method == '3sigma':
        ucl = mean + 3 * std
        lcl = mean - 3 * std
    elif method == 'percentile':
        ucl = np.percentile(data, 99.7)
        lcl = np.percentile(data, 0.3)
    else:
        raise ValueError(f"不支持的控制限计算方法: {method}")
    
    return {
        'UCL': ucl,
        'LCL': lcl,
        'CL': mean,
        'mean': mean,
        'std': std
    }


def detect_violations(data: np.ndarray, control_limits: Dict[str, float]) -> Dict[str, List[int]]:
    """
    检测控制限违反
    
    Args:
        data: 数据数组
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
    for i in range(len(data)):
        if data[i] > ucl:
            violations['above_ucl'].append(i)
        elif data[i] < lcl:
            violations['below_lcl'].append(i)
    
    # 检测趋势（连续8个点在中心线同一侧）
    cl = control_limits['CL']
    for i in range(len(data) - 7):
        segment = data[i:i+8]
        if np.all(segment > cl) or np.all(segment < cl):
            violations['trends'].append(i)
    
    return violations


def format_statistics(stats: Dict[str, Any]) -> str:
    """
    格式化统计信息
    
    Args:
        stats: 统计信息字典
        
    Returns:
        格式化的统计信息字符串
    """
    if not stats:
        return "无统计信息"
    
    lines = []
    lines.append("=== 统计信息 ===")
    
    for key, value in stats.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        elif isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    lines.append(f"  {sub_key}: {sub_value:.4f}")
                else:
                    lines.append(f"  {sub_key}: {sub_value}")
        else:
            lines.append(f"{key}: {value}")
    
    return "\n".join(lines)


def export_report(results: Dict[str, Any], filename: str = None) -> str:
    """
    导出分析报告
    
    Args:
        results: 分析结果字典
        filename: 文件名，如果为None则自动生成
        
    Returns:
        导出的文件路径
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qc_report_{timestamp}.json"
    
    # 确保输出目录存在
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    
    # 转换numpy数组为列表以便JSON序列化
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    # 转换结果
    converted_results = convert_numpy(results)
    
    # 添加报告元数据
    report_data = {
        'metadata': {
            'generated_at': datetime.datetime.now().isoformat(),
            'version': '1.0.0',
            'system': '质量控制系统'
        },
        'results': converted_results
    }
    
    # 保存报告
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    return str(filepath)


def validate_parameters(ewma_lambda: float = None, 
                       ewma_k: float = None,
                       cusum_k: float = None,
                       cusum_h: float = None) -> List[str]:
    """
    验证参数有效性
    
    Args:
        ewma_lambda: EWMA平滑参数
        ewma_k: EWMA控制限系数
        cusum_k: CUSUM参考值
        cusum_h: CUSUM决策区间
        
    Returns:
        错误信息列表
    """
    errors = []
    
    if ewma_lambda is not None:
        if not 0 <= ewma_lambda <= 1:
            errors.append("EWMA λ参数必须在[0,1]范围内")
    
    if ewma_k is not None:
        if ewma_k <= 0:
            errors.append("EWMA k参数必须大于0")
    
    if cusum_k is not None:
        if cusum_k <= 0:
            errors.append("CUSUM k参数必须大于0")
    
    if cusum_h is not None:
        if cusum_h <= 0:
            errors.append("CUSUM h参数必须大于0")
    
    return errors


def create_summary_table(ewma_stats: Dict[str, Any], 
                        cusum_stats: Dict[str, Any],
                        capability_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建汇总表格
    
    Args:
        ewma_stats: EWMA统计信息
        cusum_stats: CUSUM统计信息
        capability_stats: 过程能力统计信息
        
    Returns:
        汇总表格数据
    """
    summary = {
        'control_charts': {
            'ewma': {
                'violations': ewma_stats.get('total_violations', 0),
                'trends': ewma_stats.get('trend_count', 0),
                'in_control': ewma_stats.get('process_in_control', True)
            },
            'cusum': {
                'violations': cusum_stats.get('total_violations', 0),
                'trends': cusum_stats.get('trend_count', 0),
                'in_control': cusum_stats.get('process_in_control', True)
            }
        },
        'capability': {
            'cp': capability_stats.get('capability_indices', {}).get('Cp', 0),
            'cpk': capability_stats.get('capability_indices', {}).get('Cpk', 0),
            'defect_rate': capability_stats.get('defect_rate', {}).get('total_defect', 0)
        },
        'overall_assessment': {
            'process_stable': True,  # 需要根据具体逻辑判断
            'capability_adequate': True,  # 需要根据具体逻辑判断
            'recommendations': []
        }
    }
    
    # 生成建议
    recommendations = []
    
    if summary['control_charts']['ewma']['violations'] > 0:
        recommendations.append("EWMA控制图检测到异常点，建议调查原因")
    
    if summary['control_charts']['cusum']['violations'] > 0:
        recommendations.append("CUSUM控制图检测到异常点，建议调查原因")
    
    if summary['capability']['cpk'] < 1.0:
        recommendations.append("过程能力不足，建议改进过程")
    
    if summary['capability']['defect_rate'] > 0.01:
        recommendations.append("不合格品率较高，建议优化过程参数")
    
    summary['overall_assessment']['recommendations'] = recommendations
    
    return summary


def calculate_performance_metrics(ewma_result: Dict[str, Any],
                                cusum_result: Dict[str, Any],
                                data: np.ndarray) -> Dict[str, float]:
    """
    计算性能指标
    
    Args:
        ewma_result: EWMA分析结果
        cusum_result: CUSUM分析结果
        data: 原始数据
        
    Returns:
        性能指标字典
    """
    metrics = {}
    
    # 检测灵敏度
    ewma_violations = len(ewma_result.get('violations', {}).get('above_ucl', [])) + \
                      len(ewma_result.get('violations', {}).get('below_lcl', []))
    cusum_violations = len(cusum_result.get('violations', {}).get('upper_violations', [])) + \
                       len(cusum_result.get('violations', {}).get('lower_violations', []))
    
    metrics['ewma_sensitivity'] = ewma_violations / len(data) if len(data) > 0 else 0
    metrics['cusum_sensitivity'] = cusum_violations / len(data) if len(data) > 0 else 0
    
    # 平均运行长度（简化计算）
    metrics['ewma_arl'] = 1 / metrics['ewma_sensitivity'] if metrics['ewma_sensitivity'] > 0 else float('inf')
    metrics['cusum_arl'] = 1 / metrics['cusum_sensitivity'] if metrics['cusum_sensitivity'] > 0 else float('inf')
    
    # 过程稳定性指标
    data_std = np.std(data, ddof=1)
    data_mean = np.mean(data)
    metrics['cv'] = data_std / data_mean if data_mean != 0 else 0
    
    return metrics 