from .imagefunc import log, fit_resize_image, tensor2pil, pil2tensor, image2mask
import torch
import numpy as np
import cv2
from PIL import Image


class ImageStitchForICImproved:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE",), 
                "mask_1": ("MASK",),
                "image_2": ("IMAGE",), 
                "mask_2": ("MASK",),
                "direction": (["auto", "top-bottom", "left-right"], {"default": "auto"}),  # 图像拼接的方向
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1}),  # 尺寸需要被整除的数值
                "border_ratio": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),  # 边框扩充比例
                "image1_scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01}),  # image1手动缩放系数
                "background_color": (["#FFFFFF", "#000000"], {"default": "#FFFFFF"}),  # 背景填充颜色
            },
        }

    DESCRIPTION = "更精细的IC拼接"
    CATEGORY = "🐟Koi-Toolkit"
    FUNCTION = "main"

    RETURN_TYPES = ("IMAGESTITCHFORICIMPROVED_DATA", "IMAGE", "MASK")
    RETURN_NAMES = ("crop_data", "image", "mask")

    def isMaskEmpty(self, mask):
        if mask is None:
            return True
        if torch.all(mask == 0):
            return True
        return False

    def hex_to_rgb(self, hex_color):
        """将HEX颜色值转换为RGB元组 (0-255)"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def hex_to_rgb_normalized(self, hex_color):
        """将HEX颜色值转换为归一化的RGB元组 (0-1)"""
        r, g, b = self.hex_to_rgb(hex_color)
        return (r / 255.0, g / 255.0, b / 255.0)
    
    def get_padding_value(self, hex_color):
        """获取用于padding的颜色值 (0-1范围内的单一值，用于灰度或白色/黑色)"""
        r, g, b = self.hex_to_rgb_normalized(hex_color)
        # 对于白色返回1.0，对于黑色返回0.0，其他颜色使用亮度值
        return (r + g + b) / 3.0
    
    def create_canvas(self, height, width, background_color='#FFFFFF'):
        """根据背景颜色创建画布"""
        r, g, b = self.hex_to_rgb_normalized(background_color)
        canvas = torch.zeros((1, height, width, 3), dtype=torch.float32)
        canvas[:, :, :, 0] = r  # Red channel
        canvas[:, :, :, 1] = g  # Green channel  
        canvas[:, :, :, 2] = b  # Blue channel
        return canvas

    def pil2mask(self, image):
        image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
        mask = torch.from_numpy(image_np)
        return 1.0 - mask

    def fill_mask_holes(self, masks):
        if masks.ndim > 3:
            regions = []
            for mask in masks:
                mask_np = np.clip(255. * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
                # 使用OpenCV填充孔洞
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                filled_mask = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
                # 进一步填充较大的孔洞
                kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel2)
                
                # 直接转换回tensor，避免使用pil2mask的反转逻辑
                region_tensor = torch.from_numpy(filled_mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                regions.append(region_tensor)
            regions_tensor = torch.cat(regions, dim=0)
            return regions_tensor
        else:
            mask_np = np.clip(255. * masks.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            # 使用OpenCV填充孔洞
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            filled_mask = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
            # 进一步填充较大的孔洞
            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel2)
            
            # 直接转换回tensor，避免使用pil2mask的反转逻辑
            region_tensor = torch.from_numpy(filled_mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            return region_tensor
        
    def fillMask(self, width, height, mask, box=(0, 0), color=0):
        bg = Image.new("L", (width, height), color)  # 创建一个 'L' 模式 (灰度) 的背景
        bg.paste(mask, box, mask)  # 将遮罩粘贴到背景上
        return bg

    def emptyImage(self, width, height, batch_size=1, color=0):
        # 分别创建 R, G, B 通道的张量
        r = torch.full([batch_size, height, width, 1], ((color >> 16) & 0xFF) / 255.0)
        g = torch.full([batch_size, height, width, 1], ((color >> 8) & 0xFF) / 255.0)
        b = torch.full([batch_size, height, width, 1], (color & 0xFF) / 255.0)
        # 将三个通道合并成一个图像张量
        return torch.cat((r, g, b), dim=-1)

    def resize_image_and_mask(self, image, mask, w, h, background_color='#FFFFFF'):
        ret_images = []
        ret_masks = []
        _mask = Image.new('L', size=(w, h), color='black')
        # 使用动态背景颜色创建图像
        bg_color = self.hex_to_rgb(background_color)
        _image = Image.new('RGB', size=(w, h), color=bg_color)

        # 调整图像大小
        if image is not None and len(image) > 0:
            for i in image:
                _image = tensor2pil(i).convert('RGB')
                _image = fit_resize_image(_image, w, h, 'fit', Image.LANCZOS, background_color)
                ret_images.append(pil2tensor(_image))

        # 调整遮罩大小
        if mask is not None and len(mask) > 0:
            for m in mask:
                _mask = tensor2pil(m).convert('L')
                _mask = fit_resize_image(_mask, w, h, 'fit', Image.LANCZOS).convert('L')
                ret_masks.append(image2mask(_mask))

        # 根据处理结果返回张量
        img_tensor = torch.cat(ret_images, dim=0) if len(ret_images) > 0 else None
        mask_tensor = torch.cat(ret_masks, dim=0) if len(ret_masks) > 0 else None
        return (img_tensor, mask_tensor)

    def main(self, image_1, mask_1, image_2, mask_2, direction, divisible_by, border_ratio, image1_scale_factor, background_color):
        # 如果mask_2为空，则为 image_2 创建一个全白遮罩
        if self.isMaskEmpty(mask_2):
            mask_2 = torch.full((1, image_2.shape[1], image_2.shape[2]), 1, dtype=torch.float32, device="cpu")

        # 获取两张图的尺寸
        _, img1_h, img1_w, _ = image_1.shape
        _, img2_h, img2_w, _ = image_2.shape

        # 保存image2的原始尺寸，确保在整个过程中不改变
        orig_img2_h, orig_img2_w = img2_h, img2_w

        # 第一步：根据mask2的非零区域调整image1的大小，但确保不超过image2
        if not self.isMaskEmpty(mask_1) and not self.isMaskEmpty(mask_2):
            mask_1 = self.fill_mask_holes(mask_1)
            mask_2 = self.fill_mask_holes(mask_2)
            mask1_area = torch.count_nonzero(mask_1).float()
            mask2_area = torch.count_nonzero(mask_2).float()

            if mask1_area > 0 and mask2_area > 0:
                # 计算面积比例的平方根作为初始缩放因子
                initial_scale_ratio = (mask2_area / mask1_area).sqrt()
                
                # 计算初始缩放后的尺寸
                temp_new_width = round((img1_w * initial_scale_ratio).item())
                temp_new_height = round((img1_h * initial_scale_ratio).item())
                
                # 计算添加边框后的最大允许尺寸（不能超过image2）
                # 考虑border_ratio，计算image1能达到的最大尺寸
                max_content_w = img2_w / (1 + 2 * border_ratio)  # 减去边框后的最大内容宽度
                max_content_h = img2_h / (1 + 2 * border_ratio)  # 减去边框后的最大内容高度
                
                # 如果按mask比例缩放后会超过限制，则进行约束
                if temp_new_width > max_content_w or temp_new_height > max_content_h:
                    # 计算受约束的缩放比例
                    width_scale = max_content_w / img1_w
                    height_scale = max_content_h / img1_h
                    constrained_scale = min(width_scale, height_scale, initial_scale_ratio.item())
                    
                    new_width = round(img1_w * constrained_scale)
                    new_height = round(img1_h * constrained_scale)
                    
                else:
                    new_width = temp_new_width
                    new_height = temp_new_height
                    log(f"Using mask-based scale ratio: {initial_scale_ratio:.3f}")

                if new_width > 0 and new_height > 0:
                    image_1, mask_1 = self.resize_image_and_mask(image_1, mask_1, new_width, new_height, background_color)
                    # 更新 image_1 尺寸
                    _, img1_h, img1_w, _ = image_1.shape

        # 第1.5步：应用用户手动缩放因子（在mask匹配之后，约束计算之前）
        if image1_scale_factor != 1.0:
            # 计算手动缩放后的新尺寸
            manual_new_width = round(img1_w * image1_scale_factor)
            manual_new_height = round(img1_h * image1_scale_factor)
            
            if manual_new_width > 0 and manual_new_height > 0:
                image_1, mask_1 = self.resize_image_and_mask(image_1, mask_1, manual_new_width, manual_new_height, background_color)
                # 更新 image_1 尺寸
                _, img1_h, img1_w, _ = image_1.shape
                log(f"Applied manual scale factor {image1_scale_factor:.2f}: resized image1 to {manual_new_width}x{manual_new_height}")

        # 第二步：计算添加边框后的image1尺寸，确保不超过image2
        # 计算指定比例的最小边框大小
        min_border_h = int(img1_h * border_ratio)
        min_border_w = int(img1_w * border_ratio)
        
        # 计算加上最小边框后的临时尺寸
        temp_h = img1_h + 2 * min_border_h
        temp_w = img1_w + 2 * min_border_w
        
        # 计算能被指定数值整除的最终目标尺寸（向上取整）
        target_h = ((temp_h + divisible_by - 1) // divisible_by) * divisible_by
        target_w = ((temp_w + divisible_by - 1) // divisible_by) * divisible_by
        
        # 最终安全检查：确保不超过image2的尺寸
        expanded_img1_h = min(target_h, img2_h)
        expanded_img1_w = min(target_w, img2_w)
        
        # 如果被限制了，记录日志
        if expanded_img1_h < target_h or expanded_img1_w < target_w:
            log(f"Final size constrained: target {target_w}x{target_h} limited to {expanded_img1_w}x{expanded_img1_h} to not exceed image2")
        
        log(f"Final image1 with border: {expanded_img1_w}x{expanded_img1_h}, image2: {img2_w}x{img2_h}")

        # 第三步：如果 direction 是 'auto'，基于"最近似正方形法则"选择拼接方向
        if direction == 'auto':
            # 左右拼接：image1在左，image2在右
            # 高度取最大值，宽度相加
            final_h_lr = max(expanded_img1_h, img2_h)
            final_w_lr = expanded_img1_w + img2_w
            aspect_ratio_lr = final_w_lr / final_h_lr
            
            # 上下拼接：image1在上，image2在下
            # 宽度取最大值，高度相加
            final_h_tb = expanded_img1_h + img2_h
            final_w_tb = max(expanded_img1_w, img2_w)
            aspect_ratio_tb = final_w_tb / final_h_tb
            
            # 计算距离正方形(1:1)的偏差
            # 使用对数来处理大于1和小于1的比例，使其对称
            lr_deviation = abs(aspect_ratio_lr - 1.0) if aspect_ratio_lr >= 1.0 else abs(1.0 / aspect_ratio_lr - 1.0)
            tb_deviation = abs(aspect_ratio_tb - 1.0) if aspect_ratio_tb >= 1.0 else abs(1.0 / aspect_ratio_tb - 1.0)
            
            # 选择更接近正方形的方向（偏差更小的）
            direction = 'left-right' if lr_deviation <= tb_deviation else 'top-bottom'
            
        # 第四步：为image1添加边框
        # 计算实际需要的总padding
        total_pad_h = expanded_img1_h - img1_h
        total_pad_w = expanded_img1_w - img1_w
        
        # 平均分配到各边
        pad_top = total_pad_h // 2
        pad_bottom = total_pad_h - pad_top
        pad_left = total_pad_w // 2
        pad_right = total_pad_w - pad_left
        
        # 应用padding到image1，使用动态背景颜色
        # 为RGB三个通道分别创建padding后的图像
        r, g, b = self.hex_to_rgb_normalized(background_color)
        
        # 分别为每个通道应用padding
        img_padding = (0, 0, pad_left, pad_right, pad_top, pad_bottom)
        image_1_r = torch.nn.functional.pad(image_1[:, :, :, 0:1], img_padding, "constant", r)
        image_1_g = torch.nn.functional.pad(image_1[:, :, :, 1:2], img_padding, "constant", g)
        image_1_b = torch.nn.functional.pad(image_1[:, :, :, 2:3], img_padding, "constant", b)
        
        # 重新组合RGB通道
        image_1 = torch.cat([image_1_r, image_1_g, image_1_b], dim=-1)
        
        # 同步padding到mask1
        if not self.isMaskEmpty(mask_1):
            mask_padding = (pad_left, pad_right, pad_top, pad_bottom)
            mask_1 = torch.nn.functional.pad(mask_1, mask_padding, "constant", 0.0)

        # 第五步：执行拼接
        # 更新image1的尺寸（添加边框后）
        _, img1_h, img1_w, _ = image_1.shape
        
        # 根据拼接方向创建最终画布
        if direction == 'left-right':
            # 左右拼接：image1在左，image2在右
            final_h = max(img1_h, img2_h)
            final_w = img1_w + img2_w
            
            # 创建指定颜色的画布
            canvas = self.create_canvas(final_h, final_w, background_color)
            
            # 放置image1（左侧，垂直居中）
            y1_offset = (final_h - img1_h) // 2
            canvas[:, y1_offset:y1_offset+img1_h, 0:img1_w, :] = image_1
            
            # 放置image2（右侧，垂直居中）
            y2_offset = (final_h - img2_h) // 2
            canvas[:, y2_offset:y2_offset+img2_h, img1_w:img1_w+img2_w, :] = image_2
            
            # 创建mask（只在image2的位置）
            mask = torch.zeros((1, final_h, final_w), dtype=torch.float32)
            mask[:, y2_offset:y2_offset+img2_h, img1_w:img1_w+img2_w] = mask_2
            
            # 记录image2在画布中的位置
            x2_pos = img1_w
            y2_pos = y2_offset
            
        else:  # top-bottom
            # 上下拼接：image1在上，image2在下
            final_h = img1_h + img2_h
            final_w = max(img1_w, img2_w)
            
            # 创建指定颜色的画布
            canvas = self.create_canvas(final_h, final_w, background_color)
            
            # 放置image1（顶部，水平居中）
            x1_offset = (final_w - img1_w) // 2
            canvas[:, 0:img1_h, x1_offset:x1_offset+img1_w, :] = image_1
            
            # 放置image2（底部，水平居中）
            x2_offset = (final_w - img2_w) // 2
            canvas[:, img1_h:img1_h+img2_h, x2_offset:x2_offset+img2_w, :] = image_2
            
            # 创建mask（只在image2的位置）
            mask = torch.zeros((1, final_h, final_w), dtype=torch.float32)
            mask[:, img1_h:img1_h+img2_h, x2_offset:x2_offset+img2_w] = mask_2
            
            # 记录image2在画布中的位置
            x2_pos = x2_offset
            y2_pos = img1_h



        # 创建精细拼接 IC 数据包
        image_stitch_data = ImageStitchForICImproved_Data(
            width=orig_img2_w,   # 原始image2宽度
            height=orig_img2_h,  # 原始image2高度
            x_offset=x2_pos,     # image2在画布中的x坐标
            y_offset=y2_pos,     # image2在画布中的y坐标
            orig_width=orig_img2_w,   # 原始 image_2 的宽度
            orig_height=orig_img2_h   # 原始 image_2 的高度
        )
        
        # 返回所有结果
        return (image_stitch_data, canvas, mask)



class ImageStitchForICImproved_Data:

    def __init__(self, width, height, x_offset, y_offset, orig_width, orig_height):
        self.width = width                # 目标区域宽度
        self.height = height              # 目标区域高度
        self.x_offset = x_offset          # X偏移量
        self.y_offset = y_offset          # Y偏移量
        self.orig_width = orig_width      # 原始图像宽度
        self.orig_height = orig_height    # 原始图像高度



class ImageStitchForICImproved_CropBack:

    def __init__(self):
        self.NODE_NAME = 'imageStitchForICImproved_CropBack'

    def crop_and_scale_as(self, image: Image, size: tuple):
        target_width, target_height = size
        ret_image = fit_resize_image(image, target_width, target_height, "crop", Image.LANCZOS)
        return ret_image
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "crop_data": ("IMAGESTITCHFORICIMPROVED_DATA",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "crop_back"
    CATEGORY = "🐟Koi-Toolkit"

    def crop_back(self, image, crop_data):
        # 获取裁切参数
        width = crop_data.width
        height = crop_data.height
        x = crop_data.x_offset
        y = crop_data.y_offset
        orig_width = crop_data.orig_width
        orig_height = crop_data.orig_height
        
        # 确保坐标不超出图像边界
        x = min(x, image.shape[2] - 1)
        y = min(y, image.shape[1] - 1)
        to_x = width + x
        to_y = height + y
        
        # 裁切图像的目标区域
        img = image[:, y:to_y, x:to_x, :]
        
        # 转换为PIL图像
        pil_image = tensor2pil(img)
        
        # 使用crop_and_scale_as函数调整到原始尺寸
        ret_image = self.crop_and_scale_as(pil_image, (orig_width, orig_height))
        
        return (pil2tensor(ret_image),)




NODE_CLASS_MAPPINGS = {
    "imageStitchForICImproved": ImageStitchForICImproved,
    "imageStitchForICImproved_CropBack": ImageStitchForICImproved_CropBack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "imageStitchForICImproved": "🐟Image Stitch For IC Improved",
    "imageStitchForICImproved_CropBack": "🐟Image Stitch For IC Improved CropBack",
}