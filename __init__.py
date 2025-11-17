from . import inpaint_stitch_simple as _inpaint_stitch_simple
from . import mask_external_rectangle as _mask_external_rectangle
from . import image_stitch_improved as _image_stitch_improved
from . import image_subtraction as _image_subtraction
from . import florence2_json_display as _florence2_json_display
from . import dashscope_chat as _dashscope_chat
from . import text_split_lines as _text_split_lines

# 合并所有子模块的节点映射，防止后导入覆盖先导入
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(getattr(_inpaint_stitch_simple, "NODE_CLASS_MAPPINGS", {}))
NODE_CLASS_MAPPINGS.update(getattr(_mask_external_rectangle, "NODE_CLASS_MAPPINGS", {}))
NODE_CLASS_MAPPINGS.update(getattr(_image_stitch_improved, "NODE_CLASS_MAPPINGS", {}))
NODE_CLASS_MAPPINGS.update(getattr(_image_subtraction, "NODE_CLASS_MAPPINGS", {}))
NODE_CLASS_MAPPINGS.update(getattr(_florence2_json_display, "NODE_CLASS_MAPPINGS", {}))
NODE_CLASS_MAPPINGS.update(getattr(_dashscope_chat, "NODE_CLASS_MAPPINGS", {}))
NODE_CLASS_MAPPINGS.update(getattr(_text_split_lines, "NODE_CLASS_MAPPINGS", {}))

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_inpaint_stitch_simple, "NODE_DISPLAY_NAME_MAPPINGS", {}))
NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mask_external_rectangle, "NODE_DISPLAY_NAME_MAPPINGS", {}))
NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_image_stitch_improved, "NODE_DISPLAY_NAME_MAPPINGS", {}))
NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_image_subtraction, "NODE_DISPLAY_NAME_MAPPINGS", {}))
NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_florence2_json_display, "NODE_DISPLAY_NAME_MAPPINGS", {}))
NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_dashscope_chat, "NODE_DISPLAY_NAME_MAPPINGS", {}))
NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_text_split_lines, "NODE_DISPLAY_NAME_MAPPINGS", {}))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]