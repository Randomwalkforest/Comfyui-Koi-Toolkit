import requests
import torch
import numpy as np
from PIL import Image
import io
import random
import math

class SearchIconFreepik:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
            },
            "optional": {
                "term": ("STRING", {"multiline": False}),
                "slug": ("STRING", {"multiline": False}),
                "page": ("INT", {"default": 1, "min": 1}),
                "per_page": ("INT", {"default": 50, "min": 1, "max": 100}),
                "order": (["relevance", "recent"],),
                "thumbnail_size": ("INT", {"default": 128, "min": 16, "max": 512}),
                "color": (["none", "gradient", "solid-black", "multicolor", "azure", "black", "blue", "chartreuse", "cyan", "gray", "green", "orange", "red", "rose", "spring-green", "violet", "white", "yellow"],),
                "shape": (["none", "outline", "fill", "lineal-color", "hand-drawn"],),
                "period": (["all", "three-months", "six-months", "one-year"],),
                "free_svg": (["all", "free", "premium"],),
                "family_id": ("INT", {"default": 0, "min": 0}),
                "second_filter": ("BOOLEAN", {"default": False}),
                "second_filter_count": ("INT", {"default": 10, "min": 1}),
            }
        }

    RETURN_TYPES = ("JSON", "INT", "IMAGE", "JSON")
    RETURN_NAMES = ("json_data", "total_results", "images", "urls")
    FUNCTION = "search_icons"
    CATEGORY = "Koi-Toolkit"

    def search_icons(self, api_key, term="", slug="", page=1, per_page=10, order="relevance", thumbnail_size=128, color="black", shape="ouline", period="all", free_svg="all", family_id=0, second_filter=True, second_filter_count=4):
        url = "https://api.freepik.com/v1/icons"
        
        headers = {
            "x-freepik-api-key": api_key,
            "Accept-Language": "en-US"
        }
        
        querystring = {
            "page": page,
            "per_page": per_page,
            "order": order,
            "thumbnail_size": str(thumbnail_size)
        }
        
        if term:
            querystring["term"] = term
        if slug:
            querystring["slug"] = slug
            
        if family_id > 0:
            querystring["family-id"] = family_id

        if color != "none":
            querystring["filters[color]"] = color
        
        if shape != "none":
            querystring["filters[shape]"] = shape
            
        if period and period != "none":
            querystring["filters[period]"] = period
            
        if free_svg and free_svg != "none":
            querystring["filters[free_svg]"] = free_svg

        try:
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()
            data = response.json()
            
            # Try to get total from meta if available
            total = 0
            if "meta" in data and "pagination" in data["meta"]:
                total = data["meta"]["pagination"].get("total", 0)
            
            # Process images
            image_tensors = []
            urls = []
            items = []
            if "data" in data:
                items = data["data"]

            if second_filter and items:
                def extract_author_key(d):
                    for k in ["author", "user", "creator", "uploader", "contributor"]:
                        v = d.get(k)
                        if isinstance(v, dict):
                            for sub in ["id", "user_id", "username", "name", "slug"]:
                                if sub in v:
                                    return f"{k}:{v[sub]}"
                        elif v is not None:
                            return f"{k}:{v}"
                    for k in ["author_id", "user_id", "creator_id", "uploader_id", "contributor_id"]:
                        if k in d:
                            return f"{k}:{d[k]}"
                    return "_unknown_author"

                def extract_type_key(d):
                    if "family_id" in d:
                        return f"family:{d['family_id']}"
                    if "family-id" in d:
                        return f"family:{d['family-id']}"
                    v = d.get("family")
                    if isinstance(v, dict):
                        for sub in ["id", "slug", "name"]:
                            if sub in v:
                                return f"family:{v[sub]}"
                    for k in ["shape", "style", "category", "type"]:
                        if k in d:
                            return f"{k}:{d[k]}"
                    if "slug" in d:
                        return f"slug:{d['slug']}"
                    return "_unknown_type"

                n = len(items)
                stride = max(1, math.ceil(n / max(1, int(second_filter_count))))
                filtered = []
                seen_authors = set()
                seen_types = set()
                start = 0
                while start < n and len(filtered) < int(second_filter_count):
                    end = min(start + stride, n)
                    segment = items[start:end]
                    cands = []
                    for it in segment:
                        akey = extract_author_key(it)
                        tkey = extract_type_key(it)
                        cands.append((it, akey, tkey))
                    g1 = [c for c in cands if c[1] not in seen_authors and c[2] not in seen_types]
                    g2 = [c for c in cands if c[2] not in seen_types]
                    g3 = [c for c in cands if c[1] not in seen_authors]
                    pick = None
                    if g1:
                        pick = random.choice(g1)
                    elif g2:
                        pick = random.choice(g2)
                    elif g3:
                        pick = random.choice(g3)
                    elif cands:
                        pick = random.choice(cands)
                    if pick is not None:
                        filtered.append(pick[0])
                        seen_authors.add(pick[1])
                        seen_types.add(pick[2])
                    start += stride
                items = filtered

            if items:
                for item in items:
                    try:
                        image_url = None
                        # Check thumbnails
                        if "thumbnails" in item:
                            for thumb in item["thumbnails"]:
                                if isinstance(thumb, dict) and "url" in thumb:
                                    image_url = thumb["url"]
                        
                        # Fallback to 'icon' -> 'url' if structure differs
                        if not image_url and "icon" in item and "url" in item["icon"]:
                            image_url = item["icon"]["url"]
                        
                        if image_url:
                            urls.append(image_url)
                            img_response = requests.get(image_url, timeout=10)
                            if img_response.status_code == 200:
                                img = Image.open(io.BytesIO(img_response.content))
                                img = img.convert("RGBA")
                                img_np = np.array(img).astype(np.float32) / 255.0
                                img_tensor = torch.from_numpy(img_np)[None,]
                                image_tensors.append(img_tensor)
                    except Exception as img_err:
                        print(f"Failed to load image for item {item.get('id', 'unknown')}: {img_err}")
                        continue
            
            if image_tensors:
                # Check sizes
                first_shape = image_tensors[0].shape
                filtered_tensors = []
                for t in image_tensors:
                    if t.shape == first_shape:
                        filtered_tensors.append(t)
                    else:
                         print(f"Skipping image with mismatched shape: {t.shape} vs {first_shape}")
                
                if filtered_tensors:
                    images = torch.cat(filtered_tensors, dim=0)
                else:
                    images = torch.zeros((1, 64, 64, 4), dtype=torch.float32)
            else:
                 images = torch.zeros((1, 64, 64, 4), dtype=torch.float32)

            return (data, total, images, urls)
            
        except Exception as e:
            print(f"Error searching icons: {e}")
            # Return empty structure on error to avoid crashing downstream
            empty_img = torch.zeros((1, 64, 64, 4), dtype=torch.float32)
            return ({"error": str(e)}, 0, empty_img, [])

NODE_CLASS_MAPPINGS = {
    "SearchIconFreepik": SearchIconFreepik
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SearchIconFreepik": "🐟 Search Icon Freepik"
}
