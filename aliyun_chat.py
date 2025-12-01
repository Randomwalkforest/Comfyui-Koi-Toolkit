import os
import json
import base64
import io
import numpy as np
from PIL import Image
import torch

try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None


class AliyunChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "qwen-plus"}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "user_message": ("STRING", {"default": "你是谁？", "multiline": True}),
                "api_key": ("STRING", {"default": ""}),
                "base_url": ("STRING", {"default": "https://dashscope.aliyuncs.com/compatible-mode/v1"}),
                "enable_thinking": ("BOOLEAN", {"default": True}),
                "show_reasoning": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("content", "reasoning")
    FUNCTION = "run"
    CATEGORY = "🐟Koi-Toolkit"

    def _get_client(self, api_key, base_url):
        key = (api_key or os.getenv("DASHSCOPE_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY 未设置，且未提供 api_key")
        if OpenAI is None:
            raise RuntimeError("openai 库未安装，请在该插件的 requirements.txt 中添加 openai 并安装")
        return OpenAI(api_key=key, base_url=base_url)

    def _aggregate_stream(self, stream_iter, include_usage):
        content_parts = []
        reasoning_parts = []
        usage_obj = None
        for event in stream_iter:
            try:
                data = event.model_dump()
            except Exception:
                try:
                    data = json.loads(str(event))
                except Exception:
                    data = {}
            choices = data.get("choices") or []
            for ch in choices:
                delta = ch.get("delta") or {}
                msg = ch.get("message") or {}
                c = delta.get("content") or msg.get("content")
                if c:
                    content_parts.append(c)
                rc = delta.get("reasoning_content") or msg.get("reasoning_content")
                if rc:
                    reasoning_parts.append(rc)
            if include_usage and (data.get("usage") is not None):
                usage_obj = data.get("usage")
        content = "".join(content_parts).strip()
        reasoning = "".join(reasoning_parts).strip()
        return content, reasoning, usage_obj or {}

    def _aggregate_non_stream(self, resp):
        try:
            data = resp.model_dump()
        except Exception:
            try:
                data = json.loads(str(resp))
            except Exception:
                data = {}
        content = ""
        reasoning = ""
        try:
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = (msg.get("content") or "").strip()
                reasoning = (msg.get("reasoning_content") or "").strip()
        except Exception:
            pass
        return content, reasoning

    def run(
        self,
        model,
        user_message,
        system_prompt="You are a helpful assistant.",
        api_key="",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        enable_thinking=True,
        show_reasoning=True,
        temperature=0.7,
        max_tokens=1024,
    ):
        client = self._get_client(api_key, base_url)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        kwargs["extra_body"] = {"enable_thinking": bool(enable_thinking)}
        result = client.chat.completions.create(**kwargs)
        content, reasoning = self._aggregate_non_stream(result)

        ui_text = content if not (show_reasoning and reasoning) else f"[Reasoning]\n{reasoning}\n\n[Answer]\n{content}"
        return {"ui": {"text": [ui_text]}, "result": (content, reasoning if show_reasoning else "")}


class AliyunVLChat(AliyunChat):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("STRING", {"default": "qwen-vl-plus"}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "user_message": ("STRING", {"default": "Describe this image.", "multiline": True}),
                "api_key": ("STRING", {"default": ""}),
                "base_url": ("STRING", {"default": "https://dashscope.aliyuncs.com/compatible-mode/v1"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("content", "reasoning")
    FUNCTION = "run"
    CATEGORY = "🐟Koi-Toolkit"

    def _image_to_base64(self, image):
        if len(image.shape) == 4:
            image = image[0]
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    def run(
        self,
        image,
        model,
        user_message,
        system_prompt="You are a helpful assistant.",
        api_key="",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.7,
        max_tokens=1024,
    ):
        client = self._get_client(api_key, base_url)
        image_url = self._image_to_base64(image)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        content_list = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": user_message}
        ]
        messages.append({"role": "user", "content": content_list})

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        result = client.chat.completions.create(**kwargs)
        content, reasoning = self._aggregate_non_stream(result)

        return {"ui": {"text": [content]}, "result": (content, reasoning)}


NODE_CLASS_MAPPINGS = {
    "AliyunChat": AliyunChat,
    "AliyunVLChat": AliyunVLChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AliyunChat": "🐟 Aliyun Chat",
    "AliyunVLChat": "🐟 Aliyun VL Chat",
}
