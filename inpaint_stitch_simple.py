import torch
import torchvision.transforms.functional as F
from PIL import Image


def rescale_i(samples, width, height, algorithm: str):
    """调整图像大小的辅助函数"""
    samples = samples.movedim(-1, 1)
    algorithm = getattr(Image, algorithm.upper())  # 例如 Image.BICUBIC
    samples_pil: Image.Image = F.to_pil_image(samples[0].cpu()).resize((width, height), algorithm)
    samples = F.to_tensor(samples_pil).unsqueeze(0)
    samples = samples.movedim(1, -1)
    return samples


def stitch_simple(canvas_image, new_image, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm):
    """
    简化版的图像拼接函数，直接将新图像拼接到画布上，不使用遮罩混合
    """
    canvas_image = canvas_image.clone()
    new_image = new_image.clone()
    
    # 调整新图像大小以匹配目标区域
    _, h, w, _ = new_image.shape
    if ctc_w > w or ctc_h > h:  # 需要放大
        resized_image = rescale_i(new_image, ctc_w, ctc_h, upscale_algorithm)
    else:  # 需要缩小
        resized_image = rescale_i(new_image, ctc_w, ctc_h, downscale_algorithm)
    
    # 直接将调整大小后的图像粘贴到画布上
    canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = resized_image
    
    # 裁剪回原始图像区域
    output_image = canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w]
    
    return output_image


class SimpleImageStitch:
    """
    简化版图像拼接节点
    
    接收stitcher对象和新图像，将新图像直接拼接到原始位置，不进行遮罩混合
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "new_image": ("IMAGE",),
            }
        }
    
    CATEGORY = "🐟Koi-Toolkit"
    DESCRIPTION = "将新图像直接拼接回原始位置，不使用遮罩混合"
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    FUNCTION = "stitch_simple_image"
    
    def stitch_simple_image(self, stitcher, new_image):
        new_image = new_image.clone()
        results = []
        
        batch_size = new_image.shape[0]
        assert len(stitcher['cropped_to_canvas_x']) == batch_size or len(stitcher['cropped_to_canvas_x']) == 1, \
            "Stitch batch size doesn't match image batch size"
        
        override = False
        if len(stitcher['cropped_to_canvas_x']) != batch_size and len(stitcher['cropped_to_canvas_x']) == 1:
            override = True
        
        for b in range(batch_size):
            one_image = new_image[b]
            one_stitcher = {}
            
            # 复制算法参数
            for key in ['downscale_algorithm', 'upscale_algorithm']:
                one_stitcher[key] = stitcher[key]
            
            # 复制坐标和图像数据
            for key in ['canvas_to_orig_x', 'canvas_to_orig_y', 'canvas_to_orig_w', 'canvas_to_orig_h', 
                       'canvas_image', 'cropped_to_canvas_x', 'cropped_to_canvas_y', 
                       'cropped_to_canvas_w', 'cropped_to_canvas_h']:
                if override:  # 一个stitcher用于多张图像
                    one_stitcher[key] = stitcher[key][0]
                else:
                    one_stitcher[key] = stitcher[key][b]
            
            one_image = one_image.unsqueeze(0)
            one_image = self.stitch_single_image(one_stitcher, one_image)
            one_image = one_image.squeeze(0)
            one_image = one_image.clone()
            results.append(one_image)
        
        result_batch = torch.stack(results, dim=0)
        
        return (result_batch,)
    
    def stitch_single_image(self, stitcher, new_image):
        downscale_algorithm = stitcher['downscale_algorithm']
        upscale_algorithm = stitcher['upscale_algorithm']
        canvas_image = stitcher['canvas_image']
        
        ctc_x = stitcher['cropped_to_canvas_x']
        ctc_y = stitcher['cropped_to_canvas_y']
        ctc_w = stitcher['cropped_to_canvas_w']
        ctc_h = stitcher['cropped_to_canvas_h']
        
        cto_x = stitcher['canvas_to_orig_x']
        cto_y = stitcher['canvas_to_orig_y']
        cto_w = stitcher['canvas_to_orig_w']
        cto_h = stitcher['canvas_to_orig_h']
        
        output_image = stitch_simple(canvas_image, new_image, ctc_x, ctc_y, ctc_w, ctc_h, 
                                   cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm)
        
        return output_image


NODE_CLASS_MAPPINGS = {
    "SimpleImageStitch": SimpleImageStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleImageStitch": "🐟 Simple Image Stitch",
}
