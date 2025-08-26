import json
from comfy.comfy_types.node_typing import IO

class Florence2JsonShow:
    """
    Florence2JsonDisplay节点 - 用于显示Florence2节点输出的坐标数据
    可以直接读取Florence2节点的JSON输出并格式化显示为可预览的JSON文本
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "florence2_data": ("JSON", {"tooltip": "来自Florence2Run节点的JSON数据输出"}),
            },
            "optional": {
                "format_style": (["pretty", "compact"], {
                    "default": "pretty",
                    "tooltip": "JSON显示格式：pretty为格式化显示，compact为紧凑显示"
                }),
                "show_coordinates": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否显示坐标信息"
                }),
                "coordinate_precision": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 6,
                    "tooltip": "坐标显示精度（小数点后位数）"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "JSON")
    RETURN_NAMES = ("formatted_json", "original_data")
    FUNCTION = "display_json"
    OUTPUT_NODE = True
    CATEGORY = "Florence2/Display"

    def format_coordinates(self, data, precision=2):
        """格式化坐标数据，统一精度"""
        if isinstance(data, list):
            return [self.format_coordinates(item, precision) for item in data]
        elif isinstance(data, dict):
            formatted = {}
            for key, value in data.items():
                if key in ['bboxes', 'box', 'quad_boxes'] and isinstance(value, list):
                    # 处理坐标数组
                    formatted[key] = [
                        [round(coord, precision) if isinstance(coord, (int, float)) else coord 
                         for coord in bbox] if isinstance(bbox, list) else bbox
                        for bbox in value
                    ]
                elif isinstance(value, (int, float)) and key in ['x', 'y', 'width', 'height']:
                    # 处理单个坐标值
                    formatted[key] = round(value, precision)
                else:
                    formatted[key] = self.format_coordinates(value, precision)
            return formatted
        elif isinstance(data, (int, float)):
            return round(data, precision)
        else:
            return data

    def display_json(self, florence2_data, format_style="pretty", show_coordinates=True, coordinate_precision=2):
        try:
            # 处理输入数据
            if florence2_data is None:
                display_data = {"message": "没有数据输入", "data": None}
            else:
                # 如果需要格式化坐标
                if show_coordinates and coordinate_precision >= 0:
                    processed_data = self.format_coordinates(florence2_data, coordinate_precision)
                else:
                    processed_data = florence2_data
                
                # 创建显示数据结构
                display_data = {
                    "florence2_detection_results": processed_data,
                    "data_info": {
                        "total_detections": len(processed_data) if isinstance(processed_data, list) else 1,
                        "coordinate_precision": coordinate_precision if show_coordinates else "disabled",
                        "format_style": format_style
                    }
                }

            # 根据格式风格生成JSON字符串
            if format_style == "pretty":
                json_str = json.dumps(display_data, ensure_ascii=False, indent=2, separators=(',', ': '))
            else:  # compact
                json_str = json.dumps(display_data, ensure_ascii=False, separators=(',', ':'))

            # 使用JSON字符串，保持换行格式
            formatted_output = json_str

            print(f"Florence2JsonDisplay: 成功格式化数据，包含 {len(processed_data) if isinstance(processed_data, list) else 1} 个检测结果")

        except Exception as e:
            error_msg = f"Florence2JsonDisplay 错误: {str(e)}"
            print(error_msg)
            formatted_output = json.dumps({
                "error": error_msg,
                "original_data": str(florence2_data)
            }, ensure_ascii=False, indent=2)

        # 返回格式化的JSON字符串和原始数据
        # UI预览通过返回字典的ui键来实现
        return {
            "ui": {"text": [formatted_output]},  # 在UI中显示文本
            "result": (formatted_output, florence2_data)  # 返回给下游节点
        }

class Florence2CoordinateExtractor:
    """
    Florence2坐标提取器 - 从Florence2数据中提取特定类型的坐标信息
    """
    
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "florence2_data": ("JSON", {"tooltip": "来自Florence2Run节点的JSON数据输出"}),
                "extract_type": (["all", "bboxes_only", "labels_only", "coordinates_with_labels"], {
                    "default": "all",
                    "tooltip": "提取类型：全部/仅边界框/仅标签/坐标与标签"
                }),
            },
            "optional": {
                "filter_label": ("STRING", {
                    "default": "",
                    "tooltip": "按标签过滤（留空表示不过滤）"
                }),
                "min_confidence": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "最小置信度阈值"
                }),
            }
        }

    RETURN_TYPES = ("JSON", "STRING")
    RETURN_NAMES = ("extracted_data", "summary")
    FUNCTION = "extract_coordinates"
    CATEGORY = "Florence2/Utils"

    def extract_coordinates(self, florence2_data, extract_type="all", filter_label="", min_confidence=0.0):
        try:
            if not florence2_data:
                return ({}, "没有输入数据")

            extracted = {}
            summary_info = []

            # 处理不同的数据结构
            if isinstance(florence2_data, list):
                # 处理bbox列表格式
                for i, item in enumerate(florence2_data):
                    if isinstance(item, dict):
                        item_data = self._extract_from_dict(item, extract_type, filter_label, min_confidence)
                        if item_data:
                            extracted[f"detection_{i}"] = item_data
                    elif isinstance(item, list) and len(item) == 4:
                        # 直接的bbox坐标
                        if extract_type in ["all", "bboxes_only", "coordinates_with_labels"]:
                            extracted[f"bbox_{i}"] = {
                                "bbox": item,
                                "x": item[0], "y": item[1], 
                                "width": item[2] - item[0], 
                                "height": item[3] - item[1]
                            }

            elif isinstance(florence2_data, dict):
                extracted = self._extract_from_dict(florence2_data, extract_type, filter_label, min_confidence)

            # 生成摘要
            total_items = len(extracted)
            summary_info.append(f"提取类型: {extract_type}")
            summary_info.append(f"总检测数: {total_items}")
            if filter_label:
                summary_info.append(f"标签过滤: {filter_label}")
            if min_confidence > 0:
                summary_info.append(f"最小置信度: {min_confidence}")

            summary = " | ".join(summary_info)
            
            return (extracted, summary)

        except Exception as e:
            error_msg = f"坐标提取错误: {str(e)}"
            return ({"error": error_msg}, error_msg)

    def _extract_from_dict(self, data, extract_type, filter_label, min_confidence):
        """从字典中提取数据"""
        result = {}
        
        # 检查是否有bboxes和labels
        if 'bboxes' in data and 'labels' in data:
            bboxes = data['bboxes']
            labels = data['labels']
            
            for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                # 标签过滤
                if filter_label and filter_label.lower() not in label.lower():
                    continue
                    
                item_key = f"item_{i}"
                
                if extract_type == "all":
                    result[item_key] = {
                        "bbox": bbox,
                        "label": label,
                        "coordinates": {
                            "x": bbox[0], "y": bbox[1],
                            "width": bbox[2] - bbox[0],
                            "height": bbox[3] - bbox[1]
                        }
                    }
                elif extract_type == "bboxes_only":
                    result[item_key] = {"bbox": bbox}
                elif extract_type == "labels_only":
                    result[item_key] = {"label": label}
                elif extract_type == "coordinates_with_labels":
                    result[item_key] = {
                        "label": label,
                        "x": bbox[0], "y": bbox[1],
                        "width": bbox[2] - bbox[0],
                        "height": bbox[3] - bbox[1]
                    }
        
        # 如果没有标准的bboxes/labels结构，尝试提取其他坐标信息
        elif extract_type == "all":
            for key, value in data.items():
                if 'box' in key.lower() or 'coord' in key.lower():
                    result[key] = value
                    
        return result

# 节点注册
NODE_CLASS_MAPPINGS = {
    "Florence2JsonShow": Florence2JsonShow,
    "Florence2CoordinateExtractor": Florence2CoordinateExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Florence2JsonShow": "Florence2 JSON Show",
    "Florence2CoordinateExtractor": "Florence2 Coordinate Extractor", 
}
