import os
import time
import random
from io import BytesIO
import numpy as np
import torch
import fitz  # PyMuPDF for SVG rasterization preview
import potrace  # bitmap tracing
from PIL import Image
import folder_paths
from nodes import SaveImage
from imagefunc import (
    pil2tensor,
    tensor2pil,
)

# # Helper conversions (Tensor <-> PIL) kept minimal
# def pil2tensor(image):
#     return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# def tensor2pil(image):
#     return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
class SVGToImage:
    """SVG字符串转图像"""  
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "SVG_String": ("STRING", {"forceInput": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_svg_to_image"
    CATEGORY = ""

    def convert_svg_to_image(self, SVG_String):

        doc = fitz.open(stream=SVG_String.encode('utf-8'), filetype="svg")
        page = doc.load_page(0)
        pix = page.get_pixmap()

        image_data = pix.tobytes("png")
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")

        return (pil2tensor(pil_image),)
    
    
class SaveSVGString:
    """保存SVG字符串到文件"""
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "SVG_String": ("STRING", {"forceInput": True}),              
                "filename_prefix": ("STRING", {"default": "ComfyUI_SVG"}),
            },
            "optional": {
                "append_timestamp": ("BOOLEAN", {"default": True}),
                "custom_output_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    CATEGORY = ""
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_svg_file"

    def generate_unique_filename(self, prefix, timestamp=False):
        if timestamp:
            timestamp_str = time.strftime("%Y%m%d%H%M%S")
            return f"{prefix}_{timestamp_str}.svg"
        else:
            return f"{prefix}.svg"

    def save_svg_file(self, SVG_String, filename_prefix="ComfyUI_SVG", append_timestamp=True, custom_output_path=""):
        
        output_path = custom_output_path if custom_output_path else self.output_dir
        os.makedirs(output_path, exist_ok=True)
        
        unique_filename = self.generate_unique_filename(f"{filename_prefix}", append_timestamp)
        final_filepath = os.path.join(output_path, unique_filename)
            
            
        with open(final_filepath, "w") as svg_file:
            svg_file.write(SVG_String)
            
            
        ui_info = {"ui": {"saved_svg": unique_filename, "path": final_filepath}}

        return ui_info




class SVGPreview(SaveImage):
    """SVG字符串预览"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "SVG_String": ("STRING", {"forceInput": True})
            }
        }

    FUNCTION = "svg_preview"
    CATEGORY = ""
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz1234567890") for x in range(5))
        self.compress_level = 4

    def svg_preview(self, SVG_String):
        doc = fitz.open(stream=SVG_String.encode('utf-8'), filetype="svg")
        page = doc.load_page(0)
        pix = page.get_pixmap()

        image_data = pix.tobytes("png")
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")

        preview = pil2tensor(pil_image)

        return self.save_images(preview, "PointPreview")





class ImageToSVG_Potracer:
    """Potracer矢量化为SVG"""
    turnpolicy_map = {
        "minority": potrace.POTRACE_TURNPOLICY_MINORITY,
        "black": potrace.POTRACE_TURNPOLICY_BLACK,
        "white": potrace.POTRACE_TURNPOLICY_WHITE,
        "left": potrace.POTRACE_TURNPOLICY_LEFT,
        "right": potrace.POTRACE_TURNPOLICY_RIGHT,
        "majority": potrace.POTRACE_TURNPOLICY_MAJORITY,
    }

    @classmethod
    def INPUT_TYPES(cls):
        policy_options = list(cls.turnpolicy_map.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("INT", {"default": 128, "min": 0, "max": 255}),
            },
            "optional": {
                "input_foreground": (["White on Black", "Black on White"], {"default": "Black on White"}),
                "turnpolicy": (policy_options, {"default": "minority"}),
                "turdsize": ("INT", {"default": 2, "min": 0}),
                "corner_threshold": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.34, "step": 0.01}),
                "zero_sharp_corners": ("BOOLEAN", {"default": False}),
                "opttolerance": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "optimize_curve": ("BOOLEAN", {"default": True}),
                "foreground_color": ("STRING", {"widget": "color", "default": "#000000"}),
                "background_color": ("STRING", {"widget": "color", "default": "#FFFFFF"}),
                "stroke_color": ("STRING", {"widget": "color", "default": "#FF0000"}),
                "stroke_width": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "vectorize"
    CATEGORY = ""

    def vectorize(self, image, threshold, turnpolicy, turdsize, corner_threshold, opttolerance,
                  input_foreground="Black on White", optimize_curve=True,
                  zero_sharp_corners=False,
                  foreground_color="#000000", background_color="#FFFFFF",
                  stroke_color="#FF0000", stroke_width=0.0):
        
        image_np = image.cpu().numpy()
        batch_svg_strings = []

        for i, single_image_np in enumerate(image_np):
            orig_width_temp, orig_height_temp = (single_image_np.shape[1], single_image_np.shape[0]) if single_image_np.ndim >= 2 else (100,100)
            svg_data_for_current_image = f'<svg width="{orig_width_temp}" height="{orig_height_temp}"><desc>Error: Processing failed before SVG generation for image {i}</desc></svg>'

            try:
                pil_img = Image.fromarray((single_image_np * 255).astype(np.uint8))
                orig_width, orig_height = pil_img.size

                if orig_width <= 0 or orig_height <= 0:
                    error_svg = f'<svg width="1" height="1"><desc>Error: Invalid image dimensions for image {i}</desc></svg>'
                    batch_svg_strings.append(error_svg)
                    continue

                threshold_norm = threshold / 255.0
                if single_image_np.ndim == 3:
                    binary_np = single_image_np[:, :, 0] < threshold_norm if single_image_np.shape[2] > 1 else single_image_np[:,:,0] < threshold_norm
                elif single_image_np.ndim == 2:
                    binary_np = single_image_np < threshold_norm
                else:
                    error_svg = f'<svg width="{orig_width}" height="{orig_height}"><desc>Error: Unexpected image dimensions for image {i}</desc></svg>'
                    batch_svg_strings.append(error_svg)
                    continue

                if input_foreground == "Black on White":
                    binary_np = ~binary_np

                if np.all(binary_np) or not np.any(binary_np):
                    skipped_svg = f'<svg width="{orig_width}" height="{orig_height}"><desc>Potracer: Skipped blank image {i}</desc></svg>'
                    batch_svg_strings.append(skipped_svg)
                    continue

                turdsize_int = int(turdsize) if turdsize is not None else 0
                policy_arg = self.turnpolicy_map.get(turnpolicy, turnpolicy)
                alphamax_value_to_use = 1.34 if zero_sharp_corners else corner_threshold
                scale = 1.0

                bm = potrace.Bitmap(binary_np)
                plist = bm.trace(
                    turdsize=turdsize_int,
                    turnpolicy=policy_arg,
                    alphamax=alphamax_value_to_use,
                    opticurve=optimize_curve,
                    opttolerance=opttolerance
                )

                scaled_width = max(1, round(orig_width * scale))
                scaled_height = max(1, round(orig_height * scale))
                svg_header = f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{scaled_width}" height="{scaled_height}" viewBox="0 0 {scaled_width} {scaled_height}">'
                svg_footer = "</svg>"
                background_rect = ""
                bg_color_lower = background_color.lower()

                if bg_color_lower != "none" and bg_color_lower != "":
                    background_rect = f'<rect width="100%" height="100%" fill="{background_color}"/>'

                scaled_stroke_width = stroke_width * scale
                stroke_attr = f'stroke="{stroke_color}" stroke-width="{scaled_stroke_width}"' if scaled_stroke_width > 0 and stroke_color.lower() != "none" else 'stroke="none"'
                fill_attr = f'fill="{foreground_color}"' if foreground_color.lower() != "none" else 'fill="none"'
                if fill_attr == 'fill="none"' and stroke_attr == 'stroke="none"':
                    fill_attr = 'fill="black"'

                all_paths_svg_parts = []
                if plist:
                    fill_rule_to_use = "evenodd"
                    for curve in plist:
                        if not (hasattr(curve, 'start_point') and hasattr(curve.start_point, 'x') and hasattr(curve.start_point, 'y')):
                            continue
                        fs = curve.start_point
                        all_paths_svg_parts.append(f"M{fs.x * scale:.2f},{fs.y * scale:.2f}")

                        if not hasattr(curve, 'segments'):
                            continue
                        for segment in curve.segments:
                            valid_segment = True
                            if not (hasattr(segment, 'is_corner') and hasattr(segment, 'end_point') and hasattr(segment.end_point, 'x') and hasattr(segment.end_point, 'y')):
                                valid_segment = False

                            if valid_segment and segment.is_corner:
                                if not (hasattr(segment, 'c') and hasattr(segment.c, 'x') and hasattr(segment.c, 'y')):
                                    valid_segment = False
                                else:
                                    c_x = segment.c.x * scale
                                    c_y = segment.c.y * scale
                                    ep_x = segment.end_point.x * scale
                                    ep_y = segment.end_point.y * scale
                                    all_paths_svg_parts.append(f"L{c_x:.2f},{c_y:.2f}L{ep_x:.2f},{ep_y:.2f}")
                            elif valid_segment:
                                if not (hasattr(segment, 'c1') and hasattr(segment.c1, 'x') and hasattr(segment.c1, 'y') and \
                                        hasattr(segment, 'c2') and hasattr(segment.c2, 'x') and hasattr(segment.c2, 'y')):
                                    valid_segment = False
                                else:
                                    c1_x = segment.c1.x * scale; c1_y = segment.c1.y * scale
                                    c2_x = segment.c2.x * scale; c2_y = segment.c2.y * scale
                                    ep_x = segment.end_point.x * scale; ep_y = segment.end_point.y * scale
                                    all_paths_svg_parts.append(f"C{c1_x:.2f},{c1_y:.2f} {c2_x:.2f},{c2_y:.2f} {ep_x:.2f},{ep_y:.2f}")
                        all_paths_svg_parts.append("Z")

                    if all_paths_svg_parts:
                        path_d_attribute = "".join(all_paths_svg_parts)
                        path_element = f'<path {stroke_attr} {fill_attr} fill-rule="{fill_rule_to_use}" d="{path_d_attribute}"/>'
                        svg_data_for_current_image = svg_header + background_rect + path_element + svg_footer
                    else:
                        svg_data_for_current_image = f'{svg_header}<desc>Potracer: Path data generation failed for image {i}</desc>{svg_footer}'
                else:
                    svg_data_for_current_image = f'{svg_header}<desc>Potracer: No paths found for image {i}</desc>{svg_footer}'

                batch_svg_strings.append(svg_data_for_current_image)

            except Exception as e:
                error_svg_content = f'<svg width="100" height="100"><desc>Error processing image {i}: {type(e).__name__} - {str(e).replace("<", "&lt;").replace(">", "&gt;")}</desc></svg>'
                batch_svg_strings.append(error_svg_content)
                continue

        output_string_joined = "\n".join(batch_svg_strings)

        return (output_string_joined,)






class ImageDesaturateEdgeBinarize:
    """将彩色像素去色为灰度，并进行边缘保护的自适应二值化。

    处理流程:
    1. 识别非灰度像素 (RGB通道最大最小差异超过 color_diff_threshold) 并按亮度公式转灰。
    2. 计算灰度图的 Sobel 梯度，生成边缘掩码 (保留锐利边缘)。
    3. 使用轻度高斯平滑得到平滑灰度，边缘处保持原灰度，其余使用平滑灰度形成用于阈值分析的图。
    4. 采用 Otsu 法估算全局阈值 (如提供 override_threshold > 0 则使用用户阈值)。
    5. 输出: 灰度图 (3 通道)、二值化图 (3 通道)、边缘掩码 (单通道 MASK)。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_diff_threshold": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.001}),
                "edge_threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "override_threshold": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("grayscale", "binarized", "edges_mask")
    FUNCTION = "process"
    CATEGORY = ""

    def _luminance(self, img):
        return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    def _gaussian_blur3(self, gray):
        # gray: [B,H,W]
        kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=gray.device) / 16.0
        k = kernel.view(1, 1, 3, 3)
        inp = gray.unsqueeze(1)
        out = torch.nn.functional.conv2d(inp, k, padding=1)
        return out.squeeze(1)

    def _sobel_edges(self, gray):
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=gray.device)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=gray.device)
        kx = sobel_x.view(1, 1, 3, 3)
        ky = sobel_y.view(1, 1, 3, 3)
        inp = gray.unsqueeze(1)
        gx = torch.nn.functional.conv2d(inp, kx, padding=1)
        gy = torch.nn.functional.conv2d(inp, ky, padding=1)
        grad = torch.sqrt(gx * gx + gy * gy).squeeze(1)
        # 归一化
        maxv = grad.max()
        if maxv > 0:
            grad = grad / maxv
        return grad

    def _otsu_threshold(self, gray):
        # gray: [B,H,W] -> flatten
        flat = gray.reshape(-1)
        # 避免空张量
        if flat.numel() == 0:
            return 0.5
        hist = torch.histc(flat, bins=256, min=0.0, max=1.0)
        total = float(flat.numel())
        p = hist / (total + 1e-12)
        bin_centers = torch.linspace(0, 1, steps=256, device=gray.device)
        w1 = torch.cumsum(p, dim=0)
        w2 = 1 - w1
        cumsum_mu = torch.cumsum(p * bin_centers, dim=0)
        mean1 = cumsum_mu / (w1 + 1e-12)
        mean_total = cumsum_mu[-1]
        mean2 = (mean_total - cumsum_mu) / (w2 + 1e-12)
        sigma = w1 * w2 * (mean1 - mean2) ** 2
        idx = torch.argmax(sigma)
        return bin_centers[idx].item()

    def process(self, image, color_diff_threshold=0.02, edge_threshold=0.2, override_threshold=-1.0):
        # image: [B,H,W,3], 0-1 float
        # Step 1: 灰度 & 仅对非灰度像素去色
        luminance = self._luminance(image)
        max_c, _ = image.max(dim=-1)
        min_c, _ = image.min(dim=-1)
        color_diff = max_c - min_c
        colored_mask = (color_diff > color_diff_threshold).float()  # [B,H,W]
        colored_mask3 = colored_mask.unsqueeze(-1)
        gray3 = luminance.unsqueeze(-1).repeat(1, 1, 1, 3)  # 保留原灰度变量（结构不动）
        # 为避免彩色/非彩色交界产生锯齿：
        # 1) 计算颜色差的连续强度并用于渐进白化
        # 2) 对二值彩色掩码做轻度高斯模糊形成软边缘混合
        # 渐进白化：颜色差越大越接近纯白，边缘区仍保留一定原色以抗锯齿
        denom = (color_diff.max() - color_diff_threshold + 1e-6)
        norm_diff = (color_diff - color_diff_threshold).clamp(min=0.0) / denom
        norm_diff = norm_diff.clamp(0.0, 1.0)  # [B,H,W]
        progressive_white = image + (1.0 - image) * norm_diff.unsqueeze(-1)  # 渐进向白

        # 软边缘掩码（对二值彩色区域做小模糊），加强过渡减少锯齿
        soft_mask = self._gaussian_blur3(colored_mask)
        soft_mask = soft_mask.clamp(0.0, 1.0) ** 0.7  # 轻微增强内部（幂次<1略扩展白域）
        soft_mask3 = soft_mask.unsqueeze(-1)

        # 最终混合：原图与渐进白图按软掩码过渡
        desat = image * (1 - soft_mask3) + progressive_white * soft_mask3

        # Step 2: 边缘检测
        desat_gray = self._luminance(desat)
        grad_norm = self._sobel_edges(desat_gray)
        edge_mask = (grad_norm >= edge_threshold).float()  # [B,H,W]

        # Step 3: 平滑 + 边缘保护
        blurred = self._gaussian_blur3(desat_gray)
        adaptive_gray = desat_gray * edge_mask + blurred * (1 - edge_mask)

        # Step 4: 阈值 (Otsu 或覆盖)
        if override_threshold > 0:
            thr = override_threshold
        else:
            thr = self._otsu_threshold(adaptive_gray)
        bin_mask = (adaptive_gray >= thr).float()  # [B,H,W]

        # 输出格式: 灰度与二值化均为 3 通道 IMAGE
        gray_img = desat_gray.unsqueeze(-1).repeat(1, 1, 1, 3)
        bin_img = bin_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        return (gray_img, bin_img, edge_mask)

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SVGToImage": SVGToImage,
    "SaveSVGString": SaveSVGString,
    "SVGPreview": SVGPreview,
    "ImageToSVG_Potracer": ImageToSVG_Potracer,
    "ImageDesaturateEdgeBinarize": ImageDesaturateEdgeBinarize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVGToImage": "SVG 转图像",
    "SaveSVGString": "保存 SVG 字符串",
    "SVGPreview": "SVG 预览",
    "ImageToSVG_Potracer": "位图转 SVG (Potrace)",
    "ImageDesaturateEdgeBinarize": "去色+边缘二值化",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]