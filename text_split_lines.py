import re

class TextSplitLines:
    """将多行文本按换行符拆分为字符串列表的简单节点。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiline_text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "skip_empty": ("BOOLEAN", {"default": True}),
                "strip_whitespace": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lines",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "split_lines"
    CATEGORY = "🐟Koi-Toolkit"

    def split_lines(self, multiline_text, skip_empty=True, strip_whitespace=True):
        lines = re.split(r'(?:\r\n|\r|\n)+', multiline_text)
        if strip_whitespace:
            lines = [line.strip() for line in lines]
        if skip_empty:
            lines = [line for line in lines if line != ""]
        return (lines,)


NODE_CLASS_MAPPINGS = {
    "TextSplitLines": TextSplitLines,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextSplitLines": "🐟 Text Split Lines",
}


