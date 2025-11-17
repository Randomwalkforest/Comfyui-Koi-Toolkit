import torch
import numpy as np
import cv2
from scipy.spatial import ConvexHull


class MaskExternalRectangle:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }
    
    CATEGORY = "🐟Koi-Toolkit"
    DESCRIPTION = "计算mask非零部分的最小外接矩形，输出矩形区域为1，其余为0"
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    
    FUNCTION = "get_external_rectangle"
    
    def minimum_area_rectangle(self, points):

        if len(points) < 3:
            # 如果点少于3个，返回轴对齐的边界框
            min_x, min_y = np.min(points, axis=0)
            max_x, max_y = np.max(points, axis=0)
            return np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
        
        try:
            # 计算凸包
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            # 使用OpenCV的minAreaRect找到最小面积外接矩形
            rect = cv2.minAreaRect(hull_points.astype(np.float32))
            box = cv2.boxPoints(rect)
            return box.astype(np.int32)
            
        except Exception:
            # 如果凸包计算失败，回退到轴对齐的边界框
            min_x, min_y = np.min(points, axis=0)
            max_x, max_y = np.max(points, axis=0)
            return np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])

    def fill_polygon_mask(self, mask_shape, polygon_points):

        mask = np.zeros(mask_shape, dtype=np.uint8)
        
        # 确保坐标在有效范围内
        polygon_points = np.clip(polygon_points, 0, [mask_shape[1]-1, mask_shape[0]-1])
        
        # 使用OpenCV填充多边形
        cv2.fillPoly(mask, [polygon_points.astype(np.int32)], 1)
        
        return mask

    def get_external_rectangle(self, mask):

        # 处理输入mask的维度
        if mask.dim() == 2:
            # 如果是2D，添加batch维度
            mask = mask.unsqueeze(0)
        
        batch_size, height, width = mask.shape
        result_masks = []
        
        for i in range(batch_size):
            current_mask = mask[i]
            
            # 找到非零像素的坐标
            nonzero_coords = torch.nonzero(current_mask, as_tuple=False)
            
            if len(nonzero_coords) == 0:
                # 如果没有非零像素，返回全零mask
                result_mask = torch.zeros_like(current_mask)
            else:
                # 转换为numpy格式 (x, y)
                points = nonzero_coords.cpu().numpy()
                # 交换坐标：torch的nonzero返回(row, col)，我们需要(x, y)即(col, row)
                points = points[:, [1, 0]]  # 从(row, col)转为(x, y)
                
                # 计算最小面积外接矩形
                rectangle_points = self.minimum_area_rectangle(points)
                
                # 创建结果mask
                result_mask_np = self.fill_polygon_mask((height, width), rectangle_points)
                result_mask = torch.from_numpy(result_mask_np.astype(np.float32))
            
            result_masks.append(result_mask)
        
        # 合并所有batch的结果
        result = torch.stack(result_masks, dim=0)
        
        # 如果输入是2D的，返回2D结果
        if result.shape[0] == 1 and len(mask.shape) == 3 and mask.shape[0] == 1:
            return (result,)
        elif result.shape[0] == 1:
            return (result.squeeze(0),)
        else:
            return (result,)


NODE_CLASS_MAPPINGS = {
    "MaskExternalRectangle": MaskExternalRectangle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskExternalRectangle": "🐟 Mask External Rectangle",
}
