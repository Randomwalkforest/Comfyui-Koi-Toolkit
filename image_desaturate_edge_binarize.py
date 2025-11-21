import torch

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
    CATEGORY = "🐟Koi-Toolkit"

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
        denom = torch.clamp(color_diff.max() - color_diff_threshold, min=1e-6)
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


NODE_CLASS_MAPPINGS = {
    "ImageDesaturateEdgeBinarize": ImageDesaturateEdgeBinarize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageDesaturateEdgeBinarize": "🐟 Image Desaturate + Edge Binarize",
}