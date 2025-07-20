# EWMA & CUSUM 质量控制系统

基于EWMA（指数加权移动平均）和CUSUM（累积和）控制图的工业过程质量控制系统软件。

## 📋 项目概述

本系统是一个完整的质量控制解决方案，集成了现代统计过程控制方法，为制造业提供实时质量监控和分析能力。系统结合了EWMA和CUSUM两种控制图的优势，提供全面的质量分析功能。

### 🎯 主要特性

- **双控制图集成**: EWMA和CUSUM控制图结合使用
- **智能异常检测**: 多种统计方法综合判断
- **实时监控**: 支持在线质量监控
- **可视化分析**: 丰富的图表和仪表板
- **统计分析**: 全面的过程能力分析
- **报告生成**: 自动生成专业报告
- **Web界面**: 用户友好的Streamlit界面

## 🏗️ 技术架构

### 核心依赖库

| 类别 | 库 | 版本 | 用途 |
|------|----|------|------|
| **数据处理** | pandas | 1.5+ | 数据操作和分析 |
| | numpy | 1.21+ | 数值计算 |
| **可视化** | matplotlib | 3.5+ | 基础图表绘制 |
| | seaborn | 0.11+ | 统计图表 |
| | plotly | 5.0+ | 交互式图表 |
| **统计分析** | scipy | 1.7+ | 统计函数 |
| | scikit-learn | 1.0+ | 机器学习算法 |
| **Web界面** | streamlit | 1.20+ | Web应用框架 |
| **文件处理** | openpyxl | 3.0+ | Excel文件读写 |

### 模块结构

```
EWMA_Xhate/
├── requirements.txt          # 依赖管理
├── README.md               # 项目文档
├── qc_system/             # 核心模块
│   ├── __init__.py
│   ├── data_processor.py   # 数据处理模块
│   ├── ewma_chart.py      # EWMA控制图
│   ├── cusum_chart.py     # CUSUM控制图
│   ├── statistics.py       # 统计分析
│   ├── visualizer.py       # 可视化模块
│   └── utils/
│       └── helpers.py      # 辅助工具
├── app.py                 # Streamlit主应用
└── example.py             # 使用示例
```

## 🔧 核心功能模块

### 1. 数据处理模块 (`data_processor.py`)

**功能特性:**
- **数据导入**: 支持Excel文件导入
- **数据清洗**: 异常值检测和处理
- **数据预处理**: 标准化和验证
- **数据验证**: 完整性检查和格式验证

**主要方法:**
```python
# 数据导入
load_data(file_path, sheet_name='Sheet1')

# 数据清洗
clean_data(data, method='iqr')

# 数据验证
validate_data(data)
```

### 2. EWMA控制图模块 (`ewma_chart.py`)

**功能特性:**
- **EWMA计算**: 指数加权移动平均
- **控制限计算**: UCL、LCL、中心线
- **异常检测**: 基于控制限的异常点识别
- **统计分析**: 过程能力指标计算

**主要方法:**
```python
# EWMA计算
calculate_ewma(data, lambda_param=0.2)

# 控制限计算
calculate_control_limits(ewma_values, sigma_multiplier=3)

# 异常检测
detect_anomalies(ewma_values, ucl, lcl)
```

### 3. CUSUM控制图模块 (`cusum_chart.py`)

**功能特性:**
- **CUSUM计算**: 累积和控制图
- **V-mask方法**: 动态控制限
- **异常检测**: 趋势变化识别
- **性能分析**: 平均运行长度计算

**主要方法:**
```python
# CUSUM计算
calculate_cusum(data, target, k=0.5, h=5)

# V-mask应用
apply_v_mask(cusum_values, h, k)

# 异常检测
detect_cusum_anomalies(cusum_values, h)
```

### 4. 统计分析模块 (`statistics.py`)

**功能特性:**
- **过程能力分析**: Cp、Cpk指标
- **趋势分析**: 线性回归和时间序列分析
- **正态性检验**: Shapiro-Wilk检验
- **自相关分析**: ACF和PACF
- **异常值检测**: 多种统计方法

**主要方法:**
```python
# 过程能力分析
process_capability_analysis(data, usl, lsl)

# 趋势分析
trend_analysis(data)

# 正态性检验
normality_test(data)
```

### 5. 可视化模块 (`visualizer.py`)

**功能特性:**
- **数据概览**: 描述性统计图表
- **控制图对比**: EWMA vs CUSUM
- **过程能力分析**: 直方图和正态分布
- **趋势分析**: 时间序列图表
- **综合仪表板**: 多图表展示

**主要方法:**
```python
# 数据概览
plot_data_overview(data)

# 控制图对比
plot_control_charts_comparison(ewma_data, cusum_data)

# 过程能力分析
plot_process_capability(data, usl, lsl)
```

### 6. 辅助工具模块 (`utils/helpers.py`)

**功能特性:**
- **示例数据生成**: 模拟质量控制数据
- **控制限计算**: 统计控制限
- **异常检测**: 多种检测算法
- **参数验证**: 输入参数检查
- **报告导出**: Excel和PDF格式

**主要方法:**
```python
# 示例数据生成
generate_sample_data(n_samples=100, mean=100, std=5)

# 控制限计算
calculate_statistical_limits(data, method='3sigma')

# 报告导出
export_report(results, filename='qc_report.xlsx')
```

## 🚀 安装和使用

### 环境要求

- **Python**: 3.7+
- **操作系统**: Windows, macOS, Linux
- **内存**: 建议4GB+
- **存储**: 建议1GB可用空间

### 快速安装

1. **克隆项目**
```bash
git clone https://github.com/your-username/EWMA_Xhate.git
cd EWMA_Xhate
```

2. **创建并激活虚拟环境 (推荐)**
```bash
uv venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate   # Windows
```

3. **安装依赖**
```bash
uv pip install -r requirements.txt
```

4. **启动应用**
```bash
streamlit run app.py
```

### 使用示例

#### 基本使用

```python
from qc_system import DataProcessor, EWMAChart, CUSUMChart, Statistics, Visualizer

# 1. 数据导入
processor = DataProcessor()
data = processor.load_data('paper.xlsx')

# 2. EWMA分析
ewma_chart = EWMAChart()
ewma_results = ewma_chart.analyze(data['value'], lambda_param=0.2)

# 3. CUSUM分析
cusum_chart = CUSUMChart()
cusum_results = cusum_chart.analyze(data['value'], target=100, k=0.5, h=5)

# 4. 统计分析
stats = Statistics()
capability_results = stats.process_capability_analysis(data['value'], usl=110, lsl=90)

# 5. 可视化
viz = Visualizer()
viz.plot_control_charts_comparison(ewma_results, cusum_results)
```

#### Web界面使用

1. 启动应用后，在浏览器中打开显示的地址
2. 选择数据输入方式：
   - 上传Excel文件
   - 使用示例数据
   - 手动输入数据
3. 配置分析参数
4. 执行分析并查看结果
5. 导出报告

## 📊 功能演示

### 数据输入方式

1. **文件上传**: 支持Excel格式文件
2. **示例数据**: 系统内置示例数据
3. **手动输入**: 通过界面输入数据

### 分析功能

1. **EWMA分析**: 
   - 参数配置（λ值）
   - 控制限计算
   - 异常点检测

2. **CUSUM分析**:
   - 参数配置（k, h值）
   - V-mask应用
   - 趋势变化检测

3. **统计分析**:
   - 过程能力指标
   - 趋势分析
   - 正态性检验

### 可视化功能

1. **控制图**: EWMA和CUSUM对比
2. **过程能力**: 直方图和正态分布
3. **趋势分析**: 时间序列图表
4. **综合仪表板**: 多图表展示

### 报告功能

1. **Excel导出**: 包含所有分析结果
2. **PDF报告**: 专业格式报告
3. **图表导出**: 高质量图表文件

## 🎛️ 参数配置

### EWMA参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lambda_param` | 0.2 | 平滑参数，范围0-1 |
| `sigma_multiplier` | 3 | 控制限倍数 |

### CUSUM参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `k` | 0.5 | 参考值，通常为0.5 |
| `h` | 5 | 决策区间，通常为4-5 |

### 统计分析参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `usl` | 110 | 上规格限 |
| `lsl` | 90 | 下规格限 |
| `confidence_level` | 0.95 | 置信水平 |

## 📈 应用场景

### 制造业质量控制

- **生产线监控**: 实时监控产品质量
- **过程优化**: 识别过程改进机会
- **质量报告**: 自动生成质量报告

### 研发和测试

- **实验数据分析**: 分析实验结果
- **性能评估**: 评估系统性能
- **趋势分析**: 识别长期趋势

### 教育和培训

- **质量控制教学**: 质量控制方法教学
- **统计分析**: 统计方法实践
- **案例研究**: 实际应用案例

## 🔍 技术特点

### 算法优势

1. **EWMA优势**:
   - 对微小变化敏感
   - 平滑噪声影响
   - 适合连续过程监控

2. **CUSUM优势**:
   - 快速检测均值偏移
   - 累积效应检测
   - 适合批量过程监控

### 系统优势

1. **模块化设计**: 易于扩展和维护
2. **用户友好**: 直观的Web界面
3. **功能完整**: 从数据输入到报告导出
4. **性能优化**: 高效的数据处理
5. **可扩展性**: 支持新功能添加

## 🤝 贡献指南

### 开发环境设置

1. **Fork项目**
2. **创建分支**: `git checkout -b feature/new-feature`
3. **提交更改**: `git commit -am 'Add new feature'`
4. **推送分支**: `git push origin feature/new-feature`
5. **创建Pull Request**

### 代码规范

- 遵循PEP 8代码规范
- 添加适当的注释和文档
- 编写单元测试
- 确保代码可读性

## 📝 更新日志

### v1.0.0 (2024-01-XX)

- ✅ 初始版本发布
- ✅ EWMA控制图实现
- ✅ CUSUM控制图实现
- ✅ 统计分析模块
- ✅ 可视化模块
- ✅ Web界面实现
- ✅ 报告导出功能

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- **项目维护者**: [Your Name]
- **邮箱**: [your.email@example.com]
- **GitHub**: [https://github.com/your-username]

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和用户。

---

**注意**: 本系统适用于工业过程质量控制，请根据实际应用场景调整参数配置。 

