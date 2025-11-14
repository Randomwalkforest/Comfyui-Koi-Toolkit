import torch
import numpy as np

class ImageSubtraction:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "mode": (["absolute_diff", "signed_diff", "threshold_diff"], {"default": "absolute_diff"}),
                "threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "normalize": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("diff_mask", "diff_image")
    FUNCTION = "subtract_images"
    CATEGORY = "🐟Koi-Toolkit"
    
    def subtract_images(self, image_a, image_b, mode="absolute_diff", threshold=0.1, normalize=True):
        """
        计算两张图片的差异
        
        参数:
        - image_a: 第一张图片 (ComfyUI IMAGE格式)
        - image_b: 第二张图片 (ComfyUI IMAGE格式) 
        - mode: 差异计算模式
        - threshold: 阈值化模式下的阈值
        - normalize: 是否归一化输出
        """
        
        # 确保两张图片尺寸相同
        if image_a.shape != image_b.shape:
            # 调整image_b的尺寸匹配image_a
            batch_size, height, width, channels = image_a.shape
            image_b = torch.nn.functional.interpolate(
                image_b.permute(0, 3, 1, 2), 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            ).permute(0, 2, 3, 1)
        
        # 转换为灰度图进行差异计算（使用标准RGB到灰度的权重）
        def rgb_to_gray(img):
            return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        
        gray_a = rgb_to_gray(image_a)
        gray_b = rgb_to_gray(image_b)
        
        # 根据模式计算差异
        if mode == "absolute_diff":
            # 绝对差异
            diff = torch.abs(gray_a - gray_b)
        elif mode == "signed_diff":
            # 有符号差异 (A - B)
            diff = gray_a - gray_b
            # 将范围从[-1,1]映射到[0,1]
            diff = (diff + 1.0) / 2.0
        elif mode == "threshold_diff":
            # 阈值化差异
            abs_diff = torch.abs(gray_a - gray_b)
            diff = (abs_diff > threshold).float()
        else:
            diff = torch.abs(gray_a - gray_b)
        
        # 归一化到[0,1]范围
        if normalize and mode != "threshold_diff":
            diff_min = diff.min()
            diff_max = diff.max()
            if diff_max > diff_min:
                diff = (diff - diff_min) / (diff_max - diff_min)
        
        # 确保值在[0,1]范围内
        diff = torch.clamp(diff, 0.0, 1.0)
        
        # 创建彩色差异图像（将灰度差异复制到RGB三个通道）
        diff_image = diff.unsqueeze(-1).repeat(1, 1, 1, 3)
        
        return (diff, diff_image)


class ImageSubtractionAdvanced:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "method": (["L1", "L2", "SSIM", "per_channel"], {"default": "L1"}),
                "output_format": (["mask_only", "heatmap", "both"], {"default": "both"}),
            },
            "optional": {
                "blur_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "gamma_correction": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("diff_mask", "diff_heatmap", "overlay")
    FUNCTION = "advanced_subtract"
    CATEGORY = "🐟Koi-Toolkit"
    
    def advanced_subtract(self, image_a, image_b, method="L1", output_format="both", 
                         blur_sigma=0.0, gamma_correction=1.0):
        """
        高级差异计算
        """
        
        # 确保尺寸匹配
        if image_a.shape != image_b.shape:
            batch_size, height, width, channels = image_a.shape
            image_b = torch.nn.functional.interpolate(
                image_b.permute(0, 3, 1, 2), 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            ).permute(0, 2, 3, 1)
        
        # 应用高斯模糊（如果需要）
        if blur_sigma > 0:
            from torch.nn.functional import conv2d
            # 创建高斯核
            kernel_size = int(6 * blur_sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            gaussian_1d = torch.exp(-x**2 / (2 * blur_sigma**2))
            gaussian_1d = gaussian_1d / gaussian_1d.sum()
            
            gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
            gaussian_2d = gaussian_2d[None, None, :, :]
            
            # 对每个通道应用模糊
            for i in range(3):
                image_a[:, :, :, i:i+1] = conv2d(
                    image_a[:, :, :, i:i+1].permute(0, 3, 1, 2),
                    gaussian_2d,
                    padding=kernel_size//2
                ).permute(0, 2, 3, 1)
                
                image_b[:, :, :, i:i+1] = conv2d(
                    image_b[:, :, :, i:i+1].permute(0, 3, 1, 2),
                    gaussian_2d,
                    padding=kernel_size//2
                ).permute(0, 2, 3, 1)
        
        # 根据方法计算差异
        if method == "L1":
            diff = torch.mean(torch.abs(image_a - image_b), dim=-1)
        elif method == "L2":
            diff = torch.sqrt(torch.mean((image_a - image_b)**2, dim=-1))
        elif method == "per_channel":
            # 分通道计算差异并取最大值
            channel_diffs = torch.abs(image_a - image_b)
            diff = torch.max(channel_diffs, dim=-1)[0]
        else:  # SSIM需要更复杂的实现，这里用简化版本
            diff = torch.mean(torch.abs(image_a - image_b), dim=-1)
        
        # 应用伽马校正
        if gamma_correction != 1.0:
            diff = torch.pow(diff, gamma_correction)
        
        # 归一化
        diff_min = diff.min()
        diff_max = diff.max()
        if diff_max > diff_min:
            diff = (diff - diff_min) / (diff_max - diff_min)
        
        diff = torch.clamp(diff, 0.0, 1.0)
        
        # 创建热力图（红色表示差异大的区域）
        heatmap = torch.zeros_like(image_a)
        heatmap[..., 0] = diff  # 红色通道
        heatmap[..., 1] = 1.0 - diff  # 绿色通道（差异小的地方为绿色）
        heatmap[..., 2] = 1.0 - diff  # 蓝色通道
        
        # 创建叠加图像
        overlay = 0.7 * image_a + 0.3 * heatmap
        overlay = torch.clamp(overlay, 0.0, 1.0)
        
        return (diff, heatmap, overlay)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageSubtraction": ImageSubtraction,
    "ImageSubtractionAdvanced": ImageSubtractionAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSubtraction": "Image Subtraction",
    "ImageSubtractionAdvanced": "Image Subtraction Advanced",
}
