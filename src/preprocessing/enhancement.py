"""
Underwater Image Enhancement Module
Provides modular preprocessing algorithms designed for underwater imagery:
1. Control: Pass-through raw RGB
2. CLAHE: Contrast Limited Adaptive Histogram Equalization in LAB/HSV space
3. Gray-World: White-balancing assumption correcting underwater color cast (red attenuation)
4. Fusion: Combined Gray-World color balance followed by adaptive contrast enhancement (CLAHE)
"""

import time
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Union


class UnderwaterEnhancer:
    def __init__(self, default_method: str = "raw"):
        self.default_method = default_method
        self.available_methods = ["raw", "clahe", "gray_world", "fusion"]

    @staticmethod
    def apply_raw(image: np.ndarray) -> np.ndarray:
        """Control method: returns unchanged RGB image."""
        return image.copy()

    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
        on the L channel in LAB color space to preserve color balance while boosting local contrast.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    @staticmethod
    def apply_gray_world(image: np.ndarray) -> np.ndarray:
        """
        Applies Gray-World White Balance algorithm.
        Compensates for underwater red-channel attenuation by scaling each channel
        so that average color across the scene is neutral gray.
        """
        img_float = image.astype(np.float32)
        mean_r = np.mean(img_float[:, :, 0]) + 1e-6
        mean_g = np.mean(img_float[:, :, 1]) + 1e-6
        mean_b = np.mean(img_float[:, :, 2]) + 1e-6

        mean_gray = (mean_r + mean_g + mean_b) / 3.0

        scale_r = mean_gray / mean_r
        scale_g = mean_gray / mean_g
        scale_b = mean_gray / mean_b

        img_float[:, :, 0] = np.clip(img_float[:, :, 0] * scale_r, 0, 255)
        img_float[:, :, 1] = np.clip(img_float[:, :, 1] * scale_g, 0, 255)
        img_float[:, :, 2] = np.clip(img_float[:, :, 2] * scale_b, 0, 255)

        return img_float.astype(np.uint8)

    @classmethod
    def apply_fusion(cls, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Dual-stage enhancement:
        Stage 1: Gray-world white balance to restore color balance and compensate red loss.
        Stage 2: LAB-space CLAHE to enhance contrast and edge definition in turbid waters.
        """
        wb_image = cls.apply_gray_world(image)
        enhanced = cls.apply_clahe(wb_image, clip_limit=clip_limit)
        return enhanced

    def enhance(
        self,
        image_input: Union[np.ndarray, Image.Image, str],
        method: str = "fusion"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Enhances input image with the specified method and returns (enhanced_rgb_array, metadata).
        """
        # Load / convert to RGB numpy array
        if isinstance(image_input, str):
            bgr = cv2.imread(image_input)
            if bgr is None:
                raise ValueError(f"Could not read image from path: {image_input}")
            img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                img_rgb = image_input.copy()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        method_clean = method.lower().strip()
        start_time = time.perf_counter()

        if method_clean in ["none", "raw", "control"]:
            enhanced = self.apply_raw(img_rgb)
            applied = "raw"
        elif method_clean == "clahe":
            enhanced = self.apply_clahe(img_rgb)
            applied = "clahe"
        elif method_clean in ["gray_world", "grayworld", "gw"]:
            enhanced = self.apply_gray_world(img_rgb)
            applied = "gray_world"
        elif method_clean in ["fusion", "dual", "combined"]:
            enhanced = self.apply_fusion(img_rgb)
            applied = "fusion"
        else:
            # Fallback to raw if unknown
            enhanced = self.apply_raw(img_rgb)
            applied = "raw (unrecognized method)"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        meta = {
            "method": applied,
            "latency_ms": round(elapsed_ms, 2),
            "original_shape": img_rgb.shape,
            "enhanced_shape": enhanced.shape,
            "pixel_min": int(np.min(enhanced)),
            "pixel_max": int(np.max(enhanced))
        }

        return enhanced, meta
