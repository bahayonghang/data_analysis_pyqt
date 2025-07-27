"""
数据模型单元测试
测试所有核心数据模型的创建、验证和序列化功能
"""

import json
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pytest

from src.models import (
    FileInfo, FileType, TimeColumnInfo, TimeColumnType, DataQualityInfo,
    AnalysisResult, AnalysisType, StatisticalSummary, CorrelationResult,
    ChartConfig, ChartType, PlotStyle, ColorPalette,
    AnalysisHistory, AnalysisStatus, UserSettings, AppLogs, LogLevel
)
from src.models.validation import ValidationUtils, SerializationUtils, ModelUtils


class TestFileInfo(unittest.TestCase):
    """FileInfo模型测试"""
    
    def setUp(self):
        """设置测试数据"""
        # 创建临时测试文件
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        self.temp_file.write(b"name,age,date\nJohn,25,2023-01-01\nJane,30,2023-01-02")
        self.temp_file.close()
        
        self.test_file_path = self.temp_file.name
    
    def tearDown(self):
        """清理测试数据"""
        Path(self.test_file_path).unlink(missing_ok=True)
    
    def test_create_from_file(self):
        """测试从文件创建FileInfo"""
        file_info = FileInfo.create_from_file(self.test_file_path)
        
        self.assertIsInstance(file_info, FileInfo)
        self.assertEqual(file_info.file_type, FileType.CSV)
        self.assertEqual(file_info.file_name, Path(self.test_file_path).name)
        self.assertTrue(file_info.file_size_bytes > 0)
        self.assertEqual(len(file_info.file_hash), 32)
        self.assertFalse(file_info.is_loaded)
    
    def test_file_type_detection(self):
        """测试文件类型检测"""
        # 测试CSV文件
        csv_info = FileInfo.create_from_file(self.test_file_path)
        self.assertEqual(csv_info.file_type, FileType.CSV)
        
        # 测试不支持的文件类型
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_txt:
            temp_txt.write(b"test content")
            temp_txt_path = temp_txt.name
        
        try:
            with self.assertRaises(ValueError):
                FileInfo.create_from_file(temp_txt_path)
        finally:
            Path(temp_txt_path).unlink(missing_ok=True)
    
    def test_data_quality_info(self):
        """测试数据质量信息"""
        quality_info = DataQualityInfo(
            total_rows=100,
            total_columns=5,
            missing_values_count=10,
            duplicate_rows_count=5,
            memory_usage_mb=2.5
        )
        
        self.assertEqual(quality_info.missing_value_percentage, 2.0)
        self.assertEqual(quality_info.duplicate_percentage, 5.0)
    
    def test_time_column_info(self):
        """测试时间列信息"""
        time_col = TimeColumnInfo(
            column_name="timestamp",
            column_type=TimeColumnType.DATETIME,
            sample_values=["2023-01-01 10:00:00", "2023-01-02 11:00:00"],
            confidence_score=0.95
        )
        
        self.assertEqual(str(time_col), "timestamp (datetime)")
        self.assertEqual(time_col.confidence_score, 0.95)
    
    def test_update_data_info(self):
        """测试更新数据信息"""
        file_info = FileInfo.create_from_file(self.test_file_path)
        
        quality_info = DataQualityInfo(
            total_rows=100,
            total_columns=3,
            missing_values_count=5,
            duplicate_rows_count=2,
            memory_usage_mb=1.5
        )
        
        time_columns = [
            TimeColumnInfo(
                column_name="date",
                column_type=TimeColumnType.DATE,
                confidence_score=0.9
            )
        ]
        
        file_info.update_data_info(
            column_names=["name", "age", "date"],
            column_types={"name": "string", "age": "integer", "date": "datetime"},
            data_quality=quality_info,
            time_columns=time_columns
        )
        
        self.assertTrue(file_info.is_loaded)
        self.assertEqual(len(file_info.column_names), 3)
        self.assertEqual(file_info.primary_time_column, "date")
        self.assertTrue(file_info.has_time_column)
    
    def test_serialization(self):
        """测试序列化"""
        file_info = FileInfo.create_from_file(self.test_file_path)
        
        # 测试转换为字典
        data_dict = file_info.to_dict()
        self.assertIsInstance(data_dict, dict)
        self.assertIn('file_path', data_dict)
        
        # 测试从字典创建
        new_file_info = FileInfo.from_dict(data_dict)
        self.assertEqual(new_file_info.file_path, file_info.file_path)
        self.assertEqual(new_file_info.file_hash, file_info.file_hash)


class TestAnalysisResult(unittest.TestCase):
    """AnalysisResult模型测试"""
    
    def test_create_analysis_result(self):
        """测试创建分析结果"""
        result = AnalysisResult(
            analysis_id="test_analysis_001",
            analysis_type=AnalysisType.DESCRIPTIVE,
            file_hash="a" * 32,
            column_names=["col1", "col2"],
            execution_time_ms=1500
        )
        
        self.assertEqual(result.analysis_type, AnalysisType.DESCRIPTIVE)
        self.assertEqual(result.execution_time_seconds, 1.5)
        self.assertEqual(result.analyzed_columns_count, 2)
        self.assertTrue(result.success)
    
    def test_statistical_summary(self):
        """测试统计摘要"""
        summary = StatisticalSummary(
            count=100,
            mean=50.0,
            median=48.0,
            std=15.0,
            min_value=10.0,
            max_value=90.0,
            q25=35.0,
            q75=65.0
        )
        
        self.assertEqual(summary.range_value, 80.0)
        self.assertEqual(summary.iqr, 30.0)
        self.assertAlmostEqual(summary.coefficient_of_variation, 0.3, places=1)
    
    def test_correlation_result(self):
        """测试相关性分析结果"""
        correlation_matrix = {
            "col1": {"col1": 1.0, "col2": 0.8},
            "col2": {"col1": 0.8, "col2": 1.0}
        }
        
        corr_result = CorrelationResult(
            method="pearson",
            correlation_matrix=correlation_matrix,
            significant_pairs=[
                {"col1": "col1", "col2": "col2", "correlation": 0.8, "p_value": 0.01}
            ]
        )
        
        self.assertEqual(corr_result.method, "pearson")
        self.assertEqual(len(corr_result.significant_pairs), 1)
    
    def test_add_results(self):
        """测试添加分析结果"""
        result = AnalysisResult(
            analysis_id="test_analysis_002",
            analysis_type=AnalysisType.CORRELATION,
            file_hash="b" * 32,
            column_names=["x", "y"],
            execution_time_ms=2000
        )
        
        # 添加统计摘要
        summary = StatisticalSummary(count=50, mean=25.0, std=5.0)
        result.add_statistical_summary("x", summary)
        
        self.assertTrue(result.has_statistical_summary)
        self.assertEqual(result.get_summary_for_column("x").mean, 25.0)
        
        # 添加相关性结果
        corr_result = CorrelationResult(
            method="spearman",
            correlation_matrix={"x": {"y": 0.7}, "y": {"x": 0.7}}
        )
        result.add_correlation_result(corr_result)
        
        self.assertTrue(result.has_correlation_analysis)
    
    def test_error_handling(self):
        """测试错误处理"""
        result = AnalysisResult(
            analysis_id="test_analysis_003",
            analysis_type=AnalysisType.OUTLIER,
            file_hash="c" * 32,
            execution_time_ms=0
        )
        
        result.mark_error("测试错误消息")
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "测试错误消息")


class TestChartConfig(unittest.TestCase):
    """ChartConfig模型测试"""
    
    def test_create_chart_config(self):
        """测试创建图表配置"""
        config = ChartConfig(
            chart_id="chart_001",
            chart_type=ChartType.LINE,
            title="测试图表",
            x_column="time",
            y_columns=["value1", "value2"]
        )
        
        self.assertEqual(config.chart_type, ChartType.LINE)
        self.assertEqual(config.title, "测试图表")
        self.assertEqual(len(config.y_columns), 2)
    
    def test_figure_size(self):
        """测试图表尺寸"""
        config = ChartConfig(
            chart_id="chart_002",
            chart_type=ChartType.BAR,
            figure_width=12.0,
            figure_height=8.0
        )
        
        self.assertEqual(config.get_figure_size(), (12.0, 8.0))
    
    def test_color_palette(self):
        """测试色彩方案"""
        config = ChartConfig(
            chart_id="chart_003",
            chart_type=ChartType.SCATTER,
            color_palette=ColorPalette.VIRIDIS
        )
        
        colors = config.get_color_list()
        self.assertIsInstance(colors, list)
        self.assertTrue(len(colors) > 0)
        
        # 测试自定义颜色
        config.custom_colors = ["#FF0000", "#00FF00", "#0000FF"]
        custom_colors = config.get_color_list()
        self.assertEqual(custom_colors, ["#FF0000", "#00FF00", "#0000FF"])
    
    def test_style_preset(self):
        """测试样式预设"""
        config = ChartConfig(
            chart_id="chart_004",
            chart_type=ChartType.HEATMAP
        )
        
        config.set_style_preset("nature")
        self.assertEqual(config.plot_style, PlotStyle.NATURE)
        self.assertEqual(config.dpi, 300)
    
    def test_matplotlib_params(self):
        """测试matplotlib参数"""
        config = ChartConfig(
            chart_id="chart_005",
            chart_type=ChartType.BOX,
            figure_width=10.0,
            figure_height=6.0,
            dpi=150
        )
        
        params = config.get_matplotlib_params()
        self.assertEqual(params['figure.figsize'], (10.0, 6.0))
        self.assertEqual(params['figure.dpi'], 150)
    
    def test_plotly_layout(self):
        """测试plotly布局"""
        config = ChartConfig(
            chart_id="chart_006",
            chart_type=ChartType.SCATTER,
            title="Plotly测试图表"
        )
        
        layout = config.get_plotly_layout()
        self.assertEqual(layout['title']['text'], "Plotly测试图表")
        self.assertIn('xaxis', layout)
        self.assertIn('yaxis', layout)


class TestDatabaseModels(unittest.TestCase):
    """数据库模型测试"""
    
    def test_analysis_history(self):
        """测试分析历史记录"""
        history = AnalysisHistory(
            analysis_id="analysis_001",
            file_hash="d" * 32,
            file_name="test.csv",
            file_path="/path/to/test.csv",
            file_size=1024,
            analysis_type="descriptive",
            total_rows=100,
            total_columns=5,
            analyzed_columns=["col1", "col2", "col3"]
        )
        
        self.assertEqual(history.status, AnalysisStatus.PENDING)
        
        # 测试标记为处理中
        history.mark_processing()
        self.assertEqual(history.status, AnalysisStatus.PROCESSING)
        
        # 测试标记为完成
        result_summary = {"mean": 25.0, "std": 5.0}
        history.mark_completed(1500, result_summary)
        self.assertEqual(history.status, AnalysisStatus.COMPLETED)
        self.assertEqual(history.execution_time_ms, 1500)
        
        # 测试标记为失败
        history.mark_failed("测试错误")
        self.assertEqual(history.status, AnalysisStatus.FAILED)
        self.assertEqual(history.error_message, "测试错误")
    
    def test_user_settings(self):
        """测试用户设置"""
        # 字符串设置
        str_setting = UserSettings(
            setting_key="theme",
            setting_value="dark",
            setting_type="string",
            category="ui"
        )
        self.assertEqual(str_setting.get_typed_value(), "dark")
        
        # 整数设置
        int_setting = UserSettings(
            setting_key="max_rows",
            setting_value="1000",
            setting_type="integer"
        )
        self.assertEqual(int_setting.get_typed_value(), 1000)
        
        # 布尔设置
        bool_setting = UserSettings(
            setting_key="auto_save",
            setting_value="true",
            setting_type="boolean"
        )
        self.assertTrue(bool_setting.get_typed_value())
        
        # JSON设置
        json_setting = UserSettings(
            setting_key="chart_config",
            setting_value='{"width": 800, "height": 600}',
            setting_type="json"
        )
        json_value = json_setting.get_typed_value()
        self.assertEqual(json_value["width"], 800)
    
    def test_app_logs(self):
        """测试应用日志"""
        log = AppLogs(
            log_level=LogLevel.ERROR,
            logger_name="test_logger",
            message="测试错误消息",
            exception_type="ValueError",
            exception_message="无效的参数"
        )
        
        self.assertTrue(log.is_error())
        self.assertFalse(log.is_warning())
        self.assertTrue(log.has_exception())


class TestValidationUtils(unittest.TestCase):
    """验证工具测试"""
    
    def test_validate_md5_hash(self):
        """测试MD5哈希验证"""
        # 有效的MD5哈希
        valid_hash = "a" * 32
        self.assertTrue(ValidationUtils.validate_md5_hash(valid_hash))
        
        # 无效的MD5哈希
        invalid_hash = "invalid_hash"
        self.assertFalse(ValidationUtils.validate_md5_hash(invalid_hash))
    
    def test_validate_column_names(self):
        """测试列名验证"""
        # 有效的列名
        valid_columns = ["col1", "col2", "col3"]
        self.assertTrue(ValidationUtils.validate_column_names(valid_columns))
        
        # 重复的列名
        duplicate_columns = ["col1", "col2", "col1"]
        self.assertFalse(ValidationUtils.validate_column_names(duplicate_columns))
        
        # 空列名
        empty_columns = []
        self.assertFalse(ValidationUtils.validate_column_names(empty_columns))
    
    def test_validate_datetime_string(self):
        """测试日期时间字符串验证"""
        # 有效的日期时间格式
        valid_datetimes = [
            "2023-01-01 10:00:00",
            "2023-01-01",
            "2023/01/01 10:00:00"
        ]
        
        for dt_str in valid_datetimes:
            self.assertTrue(ValidationUtils.validate_datetime_string(dt_str))
        
        # 无效的日期时间格式
        invalid_datetime = "not_a_date"
        self.assertFalse(ValidationUtils.validate_datetime_string(invalid_datetime))


class TestSerializationUtils(unittest.TestCase):
    """序列化工具测试"""
    
    def test_datetime_serialization(self):
        """测试日期时间序列化"""
        dt = datetime(2023, 1, 1, 10, 0, 0)
        
        # 序列化
        serialized = SerializationUtils.serialize_datetime(dt)
        self.assertIsInstance(serialized, str)
        
        # 反序列化
        deserialized = SerializationUtils.deserialize_datetime(serialized)
        self.assertEqual(deserialized, dt)
    
    def test_json_serialization(self):
        """测试JSON序列化"""
        data = {
            "string": "test",
            "number": 123,
            "datetime": datetime(2023, 1, 1),
            "nested": {"key": "value"}
        }
        
        # 序列化
        json_str = SerializationUtils.safe_json_serialize(data)
        self.assertIsInstance(json_str, str)
        
        # 反序列化
        deserialized = SerializationUtils.safe_json_deserialize(json_str)
        self.assertEqual(deserialized["string"], "test")
        self.assertEqual(deserialized["number"], 123)


class TestModelUtils(unittest.TestCase):
    """模型工具测试"""
    
    def test_model_validation(self):
        """测试模型验证"""
        # 有效数据
        valid_data = {
            "analysis_id": "test_analysis_001",
            "analysis_type": "descriptive",
            "file_hash": "a" * 32,
            "execution_time_ms": 1000
        }
        
        result = ModelUtils.safe_model_validate(AnalysisResult, valid_data)
        self.assertIsInstance(result, AnalysisResult)
        
        # 无效数据
        invalid_data = {
            "analysis_id": "short",  # 太短
            "analysis_type": "invalid_type",
            "file_hash": "invalid_hash"
        }
        
        with self.assertRaises(Exception):
            ModelUtils.safe_model_validate(AnalysisResult, invalid_data)
    
    def test_model_serialization(self):
        """测试模型序列化"""
        result = AnalysisResult(
            analysis_id="test_analysis_001",
            analysis_type=AnalysisType.DESCRIPTIVE,
            file_hash="a" * 32,
            execution_time_ms=1000
        )
        
        # 转换为字典
        data_dict = ModelUtils.safe_model_dump(result)
        self.assertIsInstance(data_dict, dict)
        
        # 转换为JSON
        json_str = ModelUtils.model_to_json(result)
        self.assertIsInstance(json_str, str)
        
        # 从JSON创建模型
        new_result = ModelUtils.model_from_json(AnalysisResult, json_str)
        self.assertEqual(new_result.analysis_id, result.analysis_id)


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)