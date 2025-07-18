"""
质量控制系统主应用程序

基于Streamlit构建的Web界面，提供EWMA和CUSUM控制图分析功能。

作者: 质量控制系统开发团队
版本: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64

# 导入自定义模块
from qc_system import DataProcessor, EWMAChart, CUSUMChart, Statistics, Visualizer
from utils.helpers import (
    generate_sample_data, validate_parameters, create_summary_table,
    calculate_performance_metrics, export_report
)

# 页面配置
st.set_page_config(
    page_title="质量控制系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置页面标题
st.title("📊 质量控制系统")
st.markdown("基于EWMA和CUSUM控制图的质量监控与分析系统")

# 侧边栏配置
st.sidebar.header("系统配置")

# 数据输入方式
data_input_method = st.sidebar.selectbox(
    "数据输入方式",
    ["自动导入paper.xlsx", "上传Excel文件", "使用示例数据", "手动输入数据"],
    index=0  # 默认选择第一个选项（自动导入paper.xlsx）
)

# 检查paper.xlsx文件是否存在（静默检查，不显示在侧边栏）
import os
paper_file_exists = os.path.exists("paper.xlsx")

# 初始化数据处理器
data_processor = DataProcessor()
data = None

# 数据加载部分
if data_input_method == "自动导入paper.xlsx":
    st.header("📁 自动导入paper.xlsx")
    
    try:
        # 自动读取paper.xlsx文件
        df = pd.read_excel("paper.xlsx")
        st.success(f"✅ 成功自动加载paper.xlsx！数据形状: {df.shape}")
        
        # 显示数据预览
        st.subheader("数据预览")
        st.dataframe(df.head())
        
        # 显示所有列名
        st.subheader("可用列")
        st.write(f"所有列: {list(df.columns)}")
        
        # 选择数值列
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            selected_column = st.selectbox("选择要分析的数值列", numeric_columns)
            data = df[selected_column].dropna().values
            st.info(f"选择了列: {selected_column}, 有效数据点: {len(data)}")
            
            # 显示选中列的基本统计信息
            if len(data) > 0:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("均值", f"{np.mean(data):.2f}")
                with col2:
                    st.metric("标准差", f"{np.std(data):.2f}")
                with col3:
                    st.metric("最小值", f"{np.min(data):.2f}")
                with col4:
                    st.metric("最大值", f"{np.max(data):.2f}")
                

        else:
            st.error("❌ 未找到数值列，请检查paper.xlsx文件格式")
            st.write("文件内容预览:")
            st.dataframe(df.head(10))
            
    except FileNotFoundError:
        st.error("❌ 未找到paper.xlsx文件，请确保文件存在于项目根目录")
    except Exception as e:
        st.error(f"❌ 文件读取错误: {str(e)}")
        st.write("请检查文件格式是否正确")

elif data_input_method == "上传Excel文件":
    st.header("📁 数据上传")
    
    uploaded_file = st.file_uploader(
        "选择Excel文件",
        type=['xlsx', 'xls'],
        help="支持.xlsx和.xls格式的Excel文件"
    )
    
    if uploaded_file is not None:
        try:
            # 读取Excel文件
            df = pd.read_excel(uploaded_file)
            st.success(f"成功加载数据！数据形状: {df.shape}")
            
            # 显示数据预览
            st.subheader("数据预览")
            st.dataframe(df.head())
            
            # 选择数值列
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                selected_column = st.selectbox("选择要分析的数值列", numeric_columns)
                data = df[selected_column].dropna().values
                st.info(f"选择了列: {selected_column}, 有效数据点: {len(data)}")
                
                
            else:
                st.error("未找到数值列，请检查数据格式")
                
        except Exception as e:
            st.error(f"文件读取错误: {str(e)}")

elif data_input_method == "使用示例数据":
    st.header("🎲 示例数据生成")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.slider("样本数量", 20, 200, 100)
        mean = st.number_input("均值", value=100.0, step=0.1)
        std = st.number_input("标准差", value=5.0, min_value=0.1, step=0.1)
    
    with col2:
        trend = st.number_input("趋势系数", value=0.0, step=0.01)
        outliers = st.checkbox("添加异常值", value=False)
    
    if st.button("生成示例数据"):
        data = generate_sample_data(n_samples, mean, std, trend, outliers)
        st.success(f"生成了 {len(data)} 个数据点")
        
        # 显示数据统计
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("均值", f"{np.mean(data):.2f}")
        with col2:
            st.metric("标准差", f"{np.std(data):.2f}")
        with col3:
            st.metric("最小值", f"{np.min(data):.2f}")
        with col4:
            st.metric("最大值", f"{np.max(data):.2f}")
        


elif data_input_method == "手动输入数据":
    st.header("✏️ 手动输入数据")
    
    data_input = st.text_area(
        "请输入数据（每行一个数值，用逗号或空格分隔）",
        height=200,
        help="例如: 100, 102, 98, 105, 99"
    )
    
    if data_input:
        try:
            # 解析输入数据
            data_lines = data_input.strip().split('\n')
            data_list = []
            for line in data_lines:
                values = line.replace(',', ' ').split()
                data_list.extend([float(v) for v in values])
            
            data = np.array(data_list)
            st.success(f"成功解析 {len(data)} 个数据点")
            

            
        except Exception as e:
            st.error(f"数据解析错误: {str(e)}")

# 参数配置
if data is not None:
    st.header("⚙️ 参数配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("EWMA参数")
        ewma_lambda = st.slider("平滑参数 λ", 0.1, 0.9, 0.2, 0.1, 
                               help="控制EWMA的平滑程度，值越小越平滑")
        ewma_k = st.number_input("控制限系数 k", value=3.0, min_value=1.0, step=0.1,
                                help="控制限的倍数，通常设为3")
    
    with col2:
        st.subheader("CUSUM参数")
        cusum_k = st.slider("参考值 k", 0.1, 1.0, 0.5, 0.1,
                           help="检测偏移的参考值，通常设为0.5")
        cusum_h = st.number_input("决策区间 h", value=5.0, min_value=1.0, step=0.1,
                                 help="决策区间，通常设为5.0")
    
    # 规格限设置
    st.subheader("规格限设置")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        usl = st.number_input("上规格限 (USL)", value=0.0, step=0.1,
                             help="上规格限，可选（设为0表示不使用）")
    with col2:
        lsl = st.number_input("下规格限 (LSL)", value=0.0, step=0.1,
                             help="下规格限，可选（设为0表示不使用）")
    with col3:
        target = st.number_input("目标值", value=float(np.mean(data)), step=0.1,
                                help="过程目标值")
    
    # 参数验证
    errors = validate_parameters(ewma_lambda, ewma_k, cusum_k, cusum_h)
    if errors:
        for error in errors:
            st.error(error)
    
    # 分析按钮
    if st.button("🚀 开始分析", type="primary"):
        with st.spinner("正在进行质量控制分析..."):
            
            # 创建分析器实例
            ewma_chart = EWMAChart(lambda_param=ewma_lambda, k=ewma_k)
            cusum_chart = CUSUMChart(k=cusum_k, h=cusum_h)
            statistics = Statistics()
            visualizer = Visualizer()
            
            # 执行EWMA分析
            ewma_result = ewma_chart.fit(data, target)
            ewma_stats = ewma_chart.get_statistics()
            
            # 执行CUSUM分析
            cusum_result = cusum_chart.fit(data, target)
            cusum_stats = cusum_chart.get_statistics()
            
            # 处理规格限（如果为0则视为未设置）
            usl_final = usl if usl != 0.0 else None
            lsl_final = lsl if lsl != 0.0 else None
            
            # 执行统计分析
            capability_result = statistics.process_capability_analysis(data, usl_final, lsl_final, target)
            trend_result = statistics.trend_analysis(data)
            normality_result = statistics.normality_test(data)
            
            # 计算性能指标
            performance_metrics = calculate_performance_metrics(ewma_result, cusum_result, data)
            
            # 创建汇总表格
            summary = create_summary_table(ewma_stats, cusum_stats, capability_result)
            
            # 显示结果
            st.header("📈 分析结果")
            
            # 创建标签页
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 控制图", "📋 统计分析", "📈 可视化", "📋 汇总报告", "💾 导出"
            ])
            
            with tab1:
                st.subheader("EWMA控制图")
                
                # 创建EWMA图表
                ewma_fig = ewma_chart.plot(data)
                st.plotly_chart(ewma_fig, use_container_width=True)
                
                # EWMA统计信息
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("违反次数", ewma_stats.get('total_violations', 0))
                with col2:
                    st.metric("趋势次数", ewma_stats.get('trend_count', 0))
                with col3:
                    st.metric("过程受控", "是" if ewma_stats.get('process_in_control', True) else "否")
                with col4:
                    st.metric("λ参数", ewma_stats.get('lambda_param', 0))
                
                st.subheader("CUSUM控制图")
                
                # 创建CUSUM图表
                cusum_fig = cusum_chart.plot(data)
                st.plotly_chart(cusum_fig, use_container_width=True)
                
                # CUSUM统计信息
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("违反次数", cusum_stats.get('total_violations', 0))
                with col2:
                    st.metric("趋势次数", cusum_stats.get('trend_count', 0))
                with col3:
                    st.metric("过程受控", "是" if cusum_stats.get('process_in_control', True) else "否")
                with col4:
                    st.metric("k参数", cusum_stats.get('k', 0))
            
            with tab2:
                st.subheader("过程能力分析")
                
                # 过程能力指数
                capability_indices = capability_result.get('capability_indices', {})
                if capability_indices:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Cp", f"{capability_indices.get('Cp', 0):.3f}")
                    with col2:
                        st.metric("Cpk", f"{capability_indices.get('Cpk', 0):.3f}")
                    with col3:
                        st.metric("Pp", f"{capability_indices.get('Pp', 0):.3f}")
                    with col4:
                        st.metric("Ppk", f"{capability_indices.get('Ppk', 0):.3f}")
                
                # 基本统计量
                basic_stats = capability_result.get('basic_stats', {})
                st.subheader("基本统计量")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("均值", f"{basic_stats.get('mean', 0):.3f}")
                    st.metric("标准差", f"{basic_stats.get('std', 0):.3f}")
                with col2:
                    st.metric("最小值", f"{basic_stats.get('min', 0):.3f}")
                    st.metric("最大值", f"{basic_stats.get('max', 0):.3f}")
                with col3:
                    st.metric("中位数", f"{basic_stats.get('median', 0):.3f}")
                    st.metric("变异系数", f"{basic_stats.get('cv', 0):.3f}")
                with col4:
                    st.metric("偏度", f"{basic_stats.get('skewness', 0):.3f}")
                    st.metric("峰度", f"{basic_stats.get('kurtosis', 0):.3f}")
                
                # 趋势分析
                st.subheader("趋势分析")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("趋势方向", trend_result.get('trend_direction', '无'))
                    st.metric("趋势强度", f"{trend_result.get('trend_strength', 0):.3f}")
                with col2:
                    st.metric("趋势显著性", "是" if trend_result.get('trend_significant', False) else "否")
                    st.metric("变化点数", len(trend_result.get('change_points', [])))
                
                # 正态性检验
                st.subheader("正态性检验")
                normality = normality_result.get('is_normal', False)
                st.metric("是否服从正态分布", "是" if normality else "否")
                
                shapiro = normality_result.get('shapiro_wilk', {})
                ks = normality_result.get('kolmogorov_smirnov', {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Shapiro-Wilk检验**")
                    st.write(f"统计量: {shapiro.get('statistic', 0):.4f}")
                    st.write(f"P值: {shapiro.get('p_value', 0):.4f}")
                with col2:
                    st.write("**Kolmogorov-Smirnov检验**")
                    st.write(f"统计量: {ks.get('statistic', 0):.4f}")
                    st.write(f"P值: {ks.get('p_value', 0):.4f}")
            
            with tab3:
                st.subheader("数据可视化")
                
                # 数据概览
                overview_fig = visualizer.plot_data_overview(data)
                st.plotly_chart(overview_fig, use_container_width=True)
                
                # 控制图对比
                comparison_fig = visualizer.plot_control_chart_comparison(
                    data, ewma_result, cusum_result
                )
                st.plotly_chart(comparison_fig, use_container_width=True)
                
                # 过程能力分析图
                capability_fig = visualizer.plot_capability_analysis(data, capability_result)
                st.plotly_chart(capability_fig, use_container_width=True)
                
                # 趋势分析图
                trend_fig = visualizer.plot_trend_analysis(data, trend_result)
                st.plotly_chart(trend_fig, use_container_width=True)
                
                # 正态性检验图
                normality_fig = visualizer.plot_normality_test(data, normality_result)
                st.plotly_chart(normality_fig, use_container_width=True)
            
            with tab4:
                st.subheader("汇总报告")
                
                # 总体评估
                st.write("### 总体评估")
                assessment = summary.get('overall_assessment', {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("过程稳定性", "稳定" if assessment.get('process_stable', True) else "不稳定")
                with col2:
                    st.metric("过程能力", "充足" if assessment.get('capability_adequate', True) else "不足")
                
                # 建议
                recommendations = assessment.get('recommendations', [])
                if recommendations:
                    st.write("### 改进建议")
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")
                else:
                    st.success("✅ 过程运行良好，无需特别改进")
                
                # 性能指标
                st.write("### 性能指标")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("EWMA灵敏度", f"{performance_metrics.get('ewma_sensitivity', 0):.3f}")
                    st.metric("EWMA平均运行长度", f"{performance_metrics.get('ewma_arl', 0):.1f}")
                with col2:
                    st.metric("CUSUM灵敏度", f"{performance_metrics.get('cusum_sensitivity', 0):.3f}")
                    st.metric("CUSUM平均运行长度", f"{performance_metrics.get('cusum_arl', 0):.1f}")
                with col3:
                    st.metric("变异系数", f"{performance_metrics.get('cv', 0):.3f}")
            
            with tab5:
                st.subheader("导出功能")
                
                # 准备导出数据
                export_data = {
                    'data': data.tolist(),
                    'parameters': {
                        'ewma_lambda': ewma_lambda,
                        'ewma_k': ewma_k,
                        'cusum_k': cusum_k,
                        'cusum_h': cusum_h,
                        'usl': usl_final,
                        'lsl': lsl_final,
                        'target': target
                    },
                    'ewma_result': ewma_result,
                    'cusum_result': cusum_result,
                    'capability_result': capability_result,
                    'trend_result': trend_result,
                    'normality_result': normality_result,
                    'performance_metrics': performance_metrics,
                    'summary': summary
                }
                
                # 导出JSON报告
                if st.button("📄 导出JSON报告"):
                    try:
                        report_path = export_report(export_data)
                        st.success(f"报告已导出到: {report_path}")
                        
                        # 提供下载链接
                        with open(report_path, 'r', encoding='utf-8') as f:
                            report_content = f.read()
                        
                        st.download_button(
                            label="📥 下载报告",
                            data=report_content,
                            file_name=f"qc_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"导出失败: {str(e)}")
                
                # 导出图表
                if st.button("📊 导出图表"):
                    try:
                        # 这里可以添加图表导出功能
                        st.info("图表导出功能开发中...")
                    except Exception as e:
                        st.error(f"图表导出失败: {str(e)}")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>质量控制系统 v1.0.0 | 基于EWMA和CUSUM控制图</p>
        <p>开发团队: 质量控制系统开发团队</p>
    </div>
    """,
    unsafe_allow_html=True
) 