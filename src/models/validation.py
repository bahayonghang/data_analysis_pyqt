"""
数据验证和序列化工具
提供模型验证、序列化和数据转换功能
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from ..utils.exceptions import DataValidationError

T = TypeVar('T', bound=BaseModel)


class ValidationUtils:
    """验证工具类"""
    
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """验证文件路径"""
        try:
            from pathlib import Path
            path = Path(file_path)
            return path.exists() and path.is_file()
        except Exception:
            return False
    
    @staticmethod
    def validate_md5_hash(hash_value: str) -> bool:
        """验证MD5哈希值格式"""
        import re
        return bool(re.match(r'^[a-f0-9]{32}$', hash_value.lower()))
    
    @staticmethod
    def validate_column_names(column_names: List[str]) -> bool:
        """验证列名列表"""
        if not column_names:
            return False
        
        # 检查是否有重复列名
        if len(column_names) != len(set(column_names)):
            return False
        
        # 检查列名是否为空或包含非法字符
        for name in column_names:
            if not name or not isinstance(name, str):
                return False
            # 基本的列名验证（可以根据需要扩展）
            if name.strip() != name or len(name.strip()) == 0:
                return False
        
        return True
    
    @staticmethod
    def validate_datetime_string(date_string: str, formats: Optional[List[str]] = None) -> bool:
        """验证日期时间字符串"""
        if not date_string:
            return False
        
        default_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d',
            '%Y/%m/%d %H:%M:%S',
            '%Y/%m/%d',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y',
            '%d-%m-%Y %H:%M:%S',
            '%d-%m-%Y',
        ]
        
        formats_to_try = formats or default_formats
        
        for fmt in formats_to_try:
            try:
                datetime.strptime(date_string, fmt)
                return True
            except ValueError:
                continue
        
        return False
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float], min_val: Optional[float] = None, max_val: Optional[float] = None) -> bool:
        """验证数值范围"""
        if not isinstance(value, (int, float)):
            return False
        
        if min_val is not None and value < min_val:
            return False
        
        if max_val is not None and value > max_val:
            return False
        
        return True


class SerializationUtils:
    """序列化工具类"""
    
    @staticmethod
    def serialize_datetime(dt: datetime) -> str:
        """序列化日期时间"""
        return dt.isoformat()
    
    @staticmethod
    def deserialize_datetime(dt_string: str) -> datetime:
        """反序列化日期时间"""
        try:
            # 尝试ISO格式
            if 'T' in dt_string:
                return datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
            else:
                # 尝试其他常见格式
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d']:
                    try:
                        return datetime.strptime(dt_string, fmt)
                    except ValueError:
                        continue
                raise ValueError(f"无法解析日期时间字符串: {dt_string}")
        except Exception as e:
            raise DataValidationError(f"日期时间反序列化失败: {e}")
    
    @staticmethod
    def serialize_numpy_types(obj: Any) -> Any:
        """序列化numpy类型"""
        try:
            import numpy as np
            
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: SerializationUtils.serialize_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [SerializationUtils.serialize_numpy_types(v) for v in obj]
            else:
                return obj
        except ImportError:
            # numpy未安装，直接返回原对象
            return obj
    
    @staticmethod
    def safe_json_serialize(obj: Any, ensure_ascii: bool = False) -> str:
        """安全的JSON序列化"""
        try:
            # 预处理numpy类型
            processed_obj = SerializationUtils.serialize_numpy_types(obj)
            
            return json.dumps(
                processed_obj,
                ensure_ascii=ensure_ascii,
                default=SerializationUtils._json_default,
                separators=(',', ':')  # 紧凑格式
            )
        except Exception as e:
            raise DataValidationError(f"JSON序列化失败: {e}")
    
    @staticmethod
    def safe_json_deserialize(json_string: str) -> Any:
        """安全的JSON反序列化"""
        try:
            return json.loads(json_string)
        except Exception as e:
            raise DataValidationError(f"JSON反序列化失败: {e}")
    
    @staticmethod
    def _json_default(obj: Any) -> Any:
        """JSON序列化默认处理器"""
        if isinstance(obj, datetime):
            return SerializationUtils.serialize_datetime(obj)
        elif hasattr(obj, 'model_dump'):
            # Pydantic模型
            return obj.model_dump()
        elif hasattr(obj, '__dict__'):
            # 其他对象
            return obj.__dict__
        else:
            return str(obj)


class ModelUtils:
    """模型工具类"""
    
    @staticmethod
    def safe_model_validate(model_class: Type[T], data: Dict[str, Any], strict: bool = True) -> T:
        """安全的模型验证"""
        try:
            if strict:
                return model_class.parse_obj(data)
            else:
                # 非严格模式，移除不存在的字段
                model_fields = set(model_class.__fields__.keys())
                filtered_data = {k: v for k, v in data.items() if k in model_fields}
                return model_class.parse_obj(filtered_data)
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                loc = '.'.join(str(x) for x in error['loc'])
                error_details.append(f"{loc}: {error['msg']}")
            
            raise DataValidationError(
                f"模型验证失败 ({model_class.__name__}): " + "; ".join(error_details),
                details={"validation_errors": e.errors()}
            )
        except Exception as e:
            raise DataValidationError(f"模型验证异常 ({model_class.__name__}): {e}")
    
    @staticmethod
    def safe_model_dump(model: BaseModel, exclude_none: bool = True, by_alias: bool = False) -> Dict[str, Any]:
        """安全的模型序列化"""
        try:
            data = model.dict(exclude_none=exclude_none, by_alias=by_alias)
            return SerializationUtils.serialize_numpy_types(data)
        except Exception as e:
            raise DataValidationError(f"模型序列化失败: {e}")
    
    @staticmethod
    def model_to_json(model: BaseModel, ensure_ascii: bool = False) -> str:
        """模型转JSON字符串"""
        try:
            data = ModelUtils.safe_model_dump(model)
            return SerializationUtils.safe_json_serialize(data, ensure_ascii=ensure_ascii)
        except Exception as e:
            raise DataValidationError(f"模型JSON序列化失败: {e}")
    
    @staticmethod
    def model_from_json(model_class: Type[T], json_string: str, strict: bool = True) -> T:
        """从JSON字符串创建模型"""
        try:
            data = SerializationUtils.safe_json_deserialize(json_string)
            return ModelUtils.safe_model_validate(model_class, data, strict=strict)
        except Exception as e:
            raise DataValidationError(f"从JSON创建模型失败: {e}")
    
    @staticmethod
    def validate_model_list(model_class: Type[T], data_list: List[Dict[str, Any]], strict: bool = True) -> List[T]:
        """验证模型列表"""
        results = []
        errors = []
        
        for i, data in enumerate(data_list):
            try:
                model = ModelUtils.safe_model_validate(model_class, data, strict=strict)
                results.append(model)
            except DataValidationError as e:
                errors.append(f"索引{i}: {e.message}")
        
        if errors:
            raise DataValidationError(
                f"模型列表验证失败: " + "; ".join(errors),
                details={"failed_indices": len(errors), "total_count": len(data_list)}
            )
        
        return results
    
    @staticmethod
    def deep_merge_models(base_model: T, update_model: T) -> T:
        """深度合并两个模型"""
        try:
            base_data = ModelUtils.safe_model_dump(base_model)
            update_data = ModelUtils.safe_model_dump(update_model, exclude_none=True)
            
            def deep_merge_dict(base_dict: Dict, update_dict: Dict) -> Dict:
                result = base_dict.copy()
                for key, value in update_dict.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = deep_merge_dict(result[key], value)
                    else:
                        result[key] = value
                return result
            
            merged_data = deep_merge_dict(base_data, update_data)
            return type(base_model).parse_obj(merged_data)
        except Exception as e:
            raise DataValidationError(f"模型合并失败: {e}")


class DataTypeUtils:
    """数据类型工具类"""
    
    @staticmethod
    def infer_column_type(sample_values: List[Any], column_name: str = "") -> str:
        """推断列的数据类型"""
        if not sample_values:
            return "unknown"
        
        # 移除None值
        non_null_values = [v for v in sample_values if v is not None]
        if not non_null_values:
            return "null"
        
        # 取前100个非空值进行类型推断
        sample = non_null_values[:100]
        
        # 检查是否为数值类型
        numeric_count = 0
        integer_count = 0
        float_count = 0
        
        for value in sample:
            try:
                if isinstance(value, (int, float)):
                    numeric_count += 1
                    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
                        integer_count += 1
                    else:
                        float_count += 1
                else:
                    # 尝试转换字符串
                    str_value = str(value).strip()
                    if str_value.isdigit() or (str_value.startswith('-') and str_value[1:].isdigit()):
                        numeric_count += 1
                        integer_count += 1
                    else:
                        try:
                            float(str_value)
                            numeric_count += 1
                            float_count += 1
                        except ValueError:
                            pass
            except (ValueError, TypeError):
                pass
        
        # 如果80%以上为数值，则认为是数值类型
        if numeric_count / len(sample) >= 0.8:
            if float_count == 0:
                return "integer"
            else:
                return "float"
        
        # 检查是否为时间类型
        datetime_count = 0
        for value in sample[:20]:  # 只检查前20个值以提高性能
            str_value = str(value).strip()
            if ValidationUtils.validate_datetime_string(str_value):
                datetime_count += 1
        
        if datetime_count / min(len(sample), 20) >= 0.6:
            return "datetime"
        
        # 检查是否为布尔类型
        bool_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}
        bool_count = 0
        for value in sample:
            str_value = str(value).strip().lower()
            if str_value in bool_values:
                bool_count += 1
        
        if bool_count / len(sample) >= 0.8:
            return "boolean"
        
        # 默认为字符串类型
        return "string"
    
    @staticmethod
    def convert_to_type(value: Any, target_type: str) -> Any:
        """将值转换为指定类型"""
        if value is None:
            return None
        
        try:
            if target_type == "integer":
                return int(float(str(value)))
            elif target_type == "float":
                return float(str(value))
            elif target_type == "boolean":
                str_value = str(value).strip().lower()
                return str_value in {'true', '1', 'yes', 'y'}
            elif target_type == "datetime":
                if isinstance(value, datetime):
                    return value
                return SerializationUtils.deserialize_datetime(str(value))
            else:
                return str(value)
        except Exception:
            # 转换失败时返回原值
            return value
    
    @staticmethod
    def detect_time_column_type(column_name: str, sample_values: List[Any]) -> Dict[str, Any]:
        """检测时间列类型"""
        result = {
            "is_time_column": False,
            "time_type": "unknown",
            "confidence": 0.0,
            "format_pattern": None,
            "sample_values": []
        }
        
        if not sample_values:
            return result
        
        column_name_lower = column_name.lower()
        
        # 基于列名的初步判断
        time_keywords = ['time', 'date', 'timestamp', 'created', 'updated', 'modified']
        tag_time_keywords = ['tagtime', 'tag_time']
        
        name_score = 0.0
        if any(keyword in column_name_lower for keyword in tag_time_keywords):
            result["time_type"] = "tagtime"
            name_score = 0.8
        elif any(keyword in column_name_lower for keyword in time_keywords):
            result["time_type"] = "datetime"
            name_score = 0.6
        
        # 基于值的判断
        non_null_values = [v for v in sample_values[:50] if v is not None]
        if not non_null_values:
            return result
        
        datetime_count = 0
        timestamp_count = 0
        
        for value in non_null_values:
            str_value = str(value).strip()
            
            # 检查是否为时间戳格式
            try:
                if str_value.isdigit() and len(str_value) >= 10:
                    # Unix时间戳
                    timestamp_count += 1
                    continue
            except ValueError:
                pass
            
            # 检查是否为日期时间格式
            if ValidationUtils.validate_datetime_string(str_value):
                datetime_count += 1
        
        total_count = len(non_null_values)
        value_score = max(datetime_count / total_count, timestamp_count / total_count)
        
        # 综合评分
        final_score = max(name_score, value_score)
        
        if final_score >= 0.6:
            result["is_time_column"] = True
            result["confidence"] = final_score
            result["sample_values"] = [str(v) for v in non_null_values[:5]]
            
            if timestamp_count > datetime_count:
                result["time_type"] = "timestamp"
            elif result["time_type"] == "unknown":
                result["time_type"] = "datetime"
        
        return result