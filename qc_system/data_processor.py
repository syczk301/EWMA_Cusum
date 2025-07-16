"""
数据处理模块

本模块负责数据的导入、清洗、预处理和验证功能。
支持Excel文件导入，数据清洗，异常值检测等。

作者: 质量控制系统开发团队
版本: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """
    数据处理类
    
    负责数据的导入、清洗、预处理和验证功能。
    """
    
    def __init__(self):
        """初始化数据处理器"""
        self.data = None
        self.cleaned_data = None
        self.metadata = {}
        
    def load_excel_data(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        从Excel文件加载数据
        
        Args:
            file_path: Excel文件路径
            sheet_name: 工作表名称，默认为第一个工作表
            
        Returns:
            加载的数据DataFrame
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        try:
            if sheet_name:
                self.data = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                self.data = pd.read_excel(file_path)
            
            # 记录元数据
            self.metadata['file_path'] = file_path
            self.metadata['sheet_name'] = sheet_name
            self.metadata['shape'] = self.data.shape
            self.metadata['columns'] = list(self.data.columns)
            
            return self.data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        except Exception as e:
            raise ValueError(f"文件格式错误: {str(e)}")
    
    def clean_data(self, 
                   remove_duplicates: bool = True,
                   handle_missing: str = 'drop',
                   remove_outliers: bool = True,
                   outlier_method: str = 'iqr') -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            remove_duplicates: 是否删除重复值
            handle_missing: 缺失值处理方式 ('drop', 'fill', 'interpolate')
            remove_outliers: 是否删除异常值
            outlier_method: 异常值检测方法 ('iqr', 'zscore', 'isolation_forest')
            
        Returns:
            清洗后的数据DataFrame
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        cleaned_data = self.data.copy()
        
        # 删除重复值
        if remove_duplicates:
            initial_rows = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            removed_duplicates = initial_rows - len(cleaned_data)
            print(f"删除了 {removed_duplicates} 个重复值")
        
        # 处理缺失值
        if handle_missing == 'drop':
            cleaned_data = cleaned_data.dropna()
        elif handle_missing == 'fill':
            cleaned_data = cleaned_data.fillna(method='ffill')
        elif handle_missing == 'interpolate':
            cleaned_data = cleaned_data.interpolate()
        
        # 删除异常值
        if remove_outliers:
            cleaned_data = self._remove_outliers(cleaned_data, outlier_method)
        
        self.cleaned_data = cleaned_data
        return cleaned_data
    
    def _remove_outliers(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        删除异常值
        
        Args:
            data: 输入数据
            method: 异常值检测方法
            
        Returns:
            删除异常值后的数据
        """
        if method == 'iqr':
            return self._remove_outliers_iqr(data)
        elif method == 'zscore':
            return self._remove_outliers_zscore(data)
        elif method == 'isolation_forest':
            return self._remove_outliers_isolation_forest(data)
        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")
    
    def _remove_outliers_iqr(self, data: pd.DataFrame) -> pd.DataFrame:
        """使用IQR方法删除异常值"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 标记异常值
            outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            data = data[~outliers]
        
        return data
    
    def _remove_outliers_zscore(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """使用Z-score方法删除异常值"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            data = data[z_scores < threshold]
        
        return data
    
    def _remove_outliers_isolation_forest(self, data: pd.DataFrame) -> pd.DataFrame:
        """使用隔离森林方法删除异常值"""
        try:
            from sklearn.ensemble import IsolationForest
            
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return data
            
            # 使用数值列进行异常值检测
            numeric_data = data[numeric_columns]
            
            # 训练隔离森林模型
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(numeric_data)
            
            # 删除异常值 (outliers == -1)
            data = data[outliers == 1]
            
        except ImportError:
            print("警告: sklearn未安装，使用IQR方法替代")
            data = self._remove_outliers_iqr(data)
        
        return data
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        获取数据信息
        
        Returns:
            包含数据信息的字典
        """
        if self.data is None:
            return {}
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object']).columns)
        }
        
        if len(info['numeric_columns']) > 0:
            info['numeric_summary'] = self.data[info['numeric_columns']].describe().to_dict()
        
        return info
    
    def validate_data_for_qc(self) -> Tuple[bool, List[str]]:
        """
        验证数据是否适合质量控制分析
        
        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        if self.data is None:
            errors.append("数据未加载")
            return False, errors
        
        # 检查是否有数值列
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            errors.append("没有找到数值列")
            return False, errors
        
        # 检查数据量
        if len(self.data) < 10:
            errors.append("数据量太少，建议至少10个数据点")
        
        # 检查是否有足够的非空值
        for col in numeric_columns:
            non_null_count = self.data[col].count()
            if non_null_count < 5:
                errors.append(f"列 {col} 的有效数据点太少")
        
        return len(errors) == 0, errors
    
    def prepare_data_for_control_chart(self, 
                                     value_column: str,
                                     group_column: Optional[str] = None,
                                     time_column: Optional[str] = None) -> pd.DataFrame:
        """
        为控制图准备数据
        
        Args:
            value_column: 数值列名
            group_column: 分组列名（用于X-bar图）
            time_column: 时间列名
            
        Returns:
            准备好的数据DataFrame
        """
        if self.cleaned_data is None:
            self.cleaned_data = self.data.copy()
        
        # 选择必要的列
        columns = [value_column]
        if group_column:
            columns.append(group_column)
        if time_column:
            columns.append(time_column)
        
        prepared_data = self.cleaned_data[columns].copy()
        
        # 确保数值列是数值类型
        prepared_data[value_column] = pd.to_numeric(prepared_data[value_column], errors='coerce')
        prepared_data = prepared_data.dropna(subset=[value_column])
        
        # 如果有时间列，确保时间格式正确
        if time_column:
            prepared_data[time_column] = pd.to_datetime(prepared_data[time_column], errors='coerce')
            prepared_data = prepared_data.dropna(subset=[time_column])
        
        return prepared_data
    
    def get_sample_statistics(self, data: pd.DataFrame, value_column: str, 
                            group_column: Optional[str] = None) -> Dict[str, Any]:
        """
        计算样本统计量
        
        Args:
            data: 数据DataFrame
            value_column: 数值列名
            group_column: 分组列名
            
        Returns:
            统计量字典
        """
        if group_column:
            # 分组统计（用于X-bar图）
            grouped = data.groupby(group_column)[value_column]
            means = grouped.mean()
            ranges = grouped.max() - grouped.min()
            stds = grouped.std()
            
            return {
                'means': means.values,
                'ranges': ranges.values,
                'stds': stds.values,
                'group_names': means.index.tolist()
            }
        else:
            # 单个值统计（用于I图）
            values = data[value_column].values
            return {
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values, ddof=1)
            } 