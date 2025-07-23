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
| **数据处理** | pandas | 2.0+ | 数据操作和分析 |
| | numpy | 1.24+ | 数值计算 |
| **可视化** | matplotlib | 3.7+ | 基础图表绘制 |
| | seaborn | 0.12+ | 统计图表 |
| | plotly | 5.15+ | 交互式图表 |
| **统计分析** | scipy | 1.11+ | 统计函数 |
| | scikit-learn | 1.3+ | 机器学习算法 |
| **Web界面** | streamlit | 1.25+ | Web应用框架 |
| **文件处理** | openpyxl | 3.1+ | Excel文件读写 |

### 模块结构

```
EWMA_Cusum/
├── requirements.txt          # 依赖管理
├── README.md               # 项目文档
├── paper.xlsx              # 示例数据文件
├── qc_system/             # 核心模块
│   ├── __init__.py
│   ├── data_processor.py   # 数据处理模块
│   ├── ewma_chart.py      # EWMA控制图
│   ├── cusum_chart.py     # CUSUM控制图
│   ├── statistics.py       # 统计分析
│   └── visualizer.py       # 可视化模块
├── utils/                 # 辅助工具
│   ├── __init__.py
│   └── helpers.py          # 辅助工具函数
└── app.py                 # Streamlit主应用
```

## 🔧 核心功能模块

### 1. 数据处理模块 (`data_processor.py`)

**功能特性:**

- **数据导入**: 支持Excel文件导入和自动加载paper.xlsx
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
fit(data, target, lambda_param=0.2)

# 控制限计算
calculate_control_limits(ewma_values, k=3)

# 异常检测
detect_violations(ewma_values, ucl, lcl)
```

### 3. CUSUM控制图模块 (`cusum_chart.py`)

**功能特性:**

- **CUSUM计算**: 累积和控制图
- **决策区间**: 基于h参数的控制限
- **异常检测**: 趋势变化识别
- **性能分析**: 平均运行长度计算

**主要方法:**

```python
# CUSUM计算
fit(data, target, k=0.5, h=5)

# 控制限应用
calculate_cusum_statistics(cusum_plus, cusum_minus, h)

# 异常检测
detect_violations(cusum_plus, cusum_minus, h)
```

### 4. 统计分析模块 (`statistics.py`)

**功能特性:**

- **过程能力分析**: Cp、Cpk、Pp、Ppk指标
- **趋势分析**: 线性回归和变化点检测
- **正态性检验**: Shapiro-Wilk和Kolmogorov-Smirnov检验
- **基本统计**: 均值、标准差、偏度、峰度

**主要方法:**

```python
# 过程能力分析
process_capability_analysis(data, usl, lsl, target)

# 趋势分析
trend_analysis(data)

# 正态性检验
normality_test(data)
```

### 5. 可视化模块 (`visualizer.py`)

**功能特性:**

- **数据概览**: 时间序列图和统计图表
- **控制图对比**: EWMA vs CUSUM
- **过程能力分析**: 直方图和正态分布拟合
- **综合仪表板**: 多图表展示

**主要方法:**

```python
# 数据概览
plot_data_overview(data)

# 控制图对比
plot_control_chart_comparison(data, ewma_result, cusum_result)

# 过程能力分析
plot_capability_analysis(data, capability_result)
```

### 6. 辅助工具模块 (`utils/helpers.py`)

**功能特性:**

- **示例数据生成**: 模拟质量控制数据
- **参数验证**: 输入参数检查
- **性能指标计算**: 控制图性能评估
- **报告导出**: JSON格式报告生成

**主要方法:**

```python
# 示例数据生成
generate_sample_data(n_samples=100, mean=100, std=5, trend=0, outliers=False)

# 参数验证
validate_parameters(ewma_lambda, ewma_k, cusum_k, cusum_h)

# 报告导出
export_report(export_data)
```

## 🚀 安装和使用

### 环境要求

- **Python**: 3.8+
- **包管理器**: uv (推荐) 或 pip
- **操作系统**: Windows, macOS, Linux
- **内存**: 建议4GB+
- **存储**: 建议1GB可用空间

> **推荐使用 uv**: uv 是一个极快的 Python 包管理器，可以显著提升依赖安装速度。如果你还没有安装 uv，请参考下面的安装说明。

### 快速安装

#### 方法一：使用 uv (推荐)

1. **安装 uv**

```bash
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **克隆项目**

```bash
git clone https://github.com/syczk301/EWMA_Cusum
cd EWMA_Cusum
```

3. **创建虚拟环境并安装依赖**

```bash
uv venv
uv pip install -r requirements.txt
```

4. **激活虚拟环境**

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

5. **启动应用**

```bash
streamlit run app.py
```

#### 方法二：使用传统 pip

1. **克隆项目**

```bash
git clone https://github.com/syczk301/EWMA_Cusum
cd EWMA_Cusum
```

2. **创建并激活虚拟环境**

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS/Linux
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **启动应用**

```bash
streamlit run app.py
```

### 🚀 使用 uv 快速开始

如果你使用 uv，可以通过以下一键命令快速开始：

```bash
# 克隆项目并进入目录
git clone https://github.com/syczk301/EWMA_Cusum && cd EWMA_Cusum

# 创建环境、安装依赖、启动应用
uv venv && uv pip install -r requirements.txt && streamlit run app.py
```

### 使用示例

#### 基本使用

```python
from qc_system import DataProcessor, EWMAChart, CUSUMChart, Statistics, Visualizer

# 1. 数据导入
processor = DataProcessor()
data = processor.load_data('paper.xlsx')

# 2. EWMA分析
ewma_chart = EWMAChart(lambda_param=0.2, k=3.0)
ewma_results = ewma_chart.fit(data['value'], target=100)

# 3. CUSUM分析
cusum_chart = CUSUMChart(k=0.5, h=5.0)
cusum_results = cusum_chart.fit(data['value'], target=100)

# 4. 统计分析
stats = Statistics()
capability_results = stats.process_capability_analysis(data['value'], usl=110, lsl=90, target=100)

# 5. 可视化
viz = Visualizer()
viz.plot_control_chart_comparison(data['value'], ewma_results, cusum_results)
```

#### Web界面使用

1. 启动应用后，在浏览器中打开显示的地址（通常是 http://localhost:8501）
2. 选择数据输入方式：
   - **自动导入paper.xlsx**: 自动加载项目中的示例数据
   - **上传Excel文件**: 上传自定义Excel文件
   - **使用示例数据**: 生成模拟数据
   - **手动输入数据**: 通过文本框输入数据
3. 配置分析参数：
   - EWMA参数：λ (平滑参数) 和 k (控制限系数)
   - CUSUM参数：k (参考值) 和 h (决策区间)
   - 规格限：USL、LSL、目标值
4. 点击"开始分析"执行分析
5. 查看分析结果：
   - **控制图**: EWMA和CUSUM控制图
   - **统计分析**: 过程能力和基本统计
   - **可视化**: 各种图表展示
   - **汇总报告**: 综合分析报告
   - **导出**: 导出JSON报告

## 📊 功能特色

### 数据输入方式

1. **自动导入**: 项目包含paper.xlsx示例数据，可直接使用
2. **文件上传**: 支持Excel格式文件上传
3. **示例数据**: 可配置参数生成模拟数据
4. **手动输入**: 支持文本输入方式

### 分析功能

1. **EWMA分析**: 
   - 可调节平滑参数λ (0.1-0.9)
   - 可调节控制限系数k
   - 自动检测违反点和趋势

2. **CUSUM分析**:
   - 可调节参考值k (0.1-1.0)
   - 可调节决策区间h
   - 累积和上下控制图

3. **统计分析**:
   - 过程能力指标 (Cp, Cpk, Pp, Ppk)
   - 基本统计量
   - 趋势分析
   - 正态性检验

### 可视化功能

1. **控制图**: 交互式EWMA和CUSUM控制图
2. **数据概览**: 时间序列图和统计图表
3. **过程能力**: 直方图和正态分布拟合
4. **趋势分析**: 回归分析图表
5. **正态性检验**: Q-Q图和直方图

### 报告功能

1. **JSON导出**: 包含所有分析结果的JSON报告
2. **下载功能**: 一键下载分析报告
3. **图表保存**: 支持图表导出（开发中）

## 🎛️ 参数配置指南

### EWMA参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `lambda_param` | 0.2 | 0.1-0.9 | 平滑参数，值越小越平滑 |
| `k` | 3.0 | 1.0+ | 控制限系数，通常设为3 |

### CUSUM参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `k` | 0.5 | 0.1-1.0 | 参考值，通常为0.5 |
| `h` | 5.0 | 1.0+ | 决策区间，通常为4-5 |

### 规格限参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `USL` | 0.0 | 上规格限，设为0表示不使用 |
| `LSL` | 0.0 | 下规格限，设为0表示不使用 |
| `target` | 数据均值 | 过程目标值 |

## 📈 应用场景

### 制造业质量控制

- **生产线监控**: 实时监控产品质量参数
- **过程优化**: 识别过程改进机会
- **质量报告**: 自动生成质量分析报告
- **异常预警**: 及时发现质量异常

### 研发和测试

- **实验数据分析**: 分析实验结果稳定性
- **性能评估**: 评估产品性能稳定性
- **质量改进**: 识别改进方向

### 教育和培训

- **质量控制教学**: SPC方法教学演示
- **统计分析**: 统计方法实践
- **案例研究**: 实际应用案例分析

## 🔍 技术特点

### 算法优势

1. **EWMA优势**:
   - 对微小变化敏感
   - 平滑随机噪声
   - 适合连续过程监控
   - 内存需求小

2. **CUSUM优势**:
   - 快速检测均值偏移
   - 累积效应检测
   - 适合检测小偏移
   - 平均运行长度短

### 系统优势

1. **模块化设计**: 易于扩展和维护
2. **用户友好**: 直观的Web界面
3. **功能完整**: 从数据输入到报告导出
4. **性能优化**: 高效的数据处理
5. **交互式**: 丰富的图表交互功能

## 🤝 贡献指南

### 开发环境设置

1. **Fork项目到你的GitHub账户**
2. **克隆Fork的项目**: `git clone https://github.com/yourusername/EWMA_Cusum`
3. **设置开发环境**:
   ```bash
   cd EWMA_Cusum
   # 使用 uv (推荐)
   pip install uv
   uv venv
   uv pip install -r requirements.txt
   
   # 或使用传统方式
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```
4. **创建功能分支**: `git checkout -b feature/new-feature`
4. **提交更改**: `git commit -am 'Add new feature'`
5. **推送分支**: `git push origin feature/new-feature`
6. **创建Pull Request**

### 代码规范

- 遵循PEP 8代码规范
- 添加适当的注释和文档字符串
- 编写单元测试
- 确保代码可读性和可维护性

### 贡献类型

- Bug修复
- 新功能开发
- 文档改进
- 性能优化
- 测试用例

## 📝 版本历史

### v1.0.0 (2025-7-22)

- ✅ 初始版本发布
- ✅ EWMA控制图实现
- ✅ CUSUM控制图实现
- ✅ 统计分析模块
- ✅ 可视化模块
- ✅ Streamlit Web界面
- ✅ 数据导入导出功能
- ✅ 报告生成功能

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- **项目维护者**: syczk301
- **GitHub**: https://github.com/syczk301/EWMA_Cusum
- **问题反馈**: 请在GitHub Issues中提交

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和用户，以及提供技术支持的开源社区。

---

**注意**: 本系统适用于工业过程质量控制，请根据实际应用场景调整参数配置。建议在正式使用前充分测试和验证。
