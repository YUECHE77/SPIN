import os
from .clip_encoder import CLIPVisionTower
from .siglip import SIGLIPVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") \
            or "intern" in vision_tower.lower():
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("bczhou"):
        return SIGLIPVisionTower(vision_tower_cfg)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
