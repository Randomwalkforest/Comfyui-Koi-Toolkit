import torch


class ImageBinarize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "binarize"
    CATEGORY = "🐟Koi-Toolkit"

    def binarize(self, image, threshold=0.5):
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        mask = (gray >= threshold).float()
        bw = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        return (bw,)


NODE_CLASS_MAPPINGS = {
    "ImageBinarize": ImageBinarize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBinarize": "🐟 Image Binarize",
}
