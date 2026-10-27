"""
Cross-Domain Generalization & Zero-Shot Underwater Evaluation Module
Evaluates trained models on out-of-distribution underwater domains
(turbid estuarine waters, open-ocean coral reefs, low-light benthic cameras).
Provides taxonomy mapping (e.g. mapping fine-grained species to generic 'fish')
and quantifies performance degradation due to domain shift.
"""

import os
import cv2
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class CrossDomainEvaluator:
    def __init__(
        self,
        model,
        target_images_dir: str,
        ground_truth_labels_dir: Optional[str] = None,
        source_domain_name: str = "Tropical Aquarium / Roboflow",
        target_domain_name: str = "Wild Aquatic / Brackish",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: Optional[str] = None
    ):
        self.model = model
        self.target_images_dir = Path(target_images_dir)
        self.ground_truth_labels_dir = Path(ground_truth_labels_dir) if ground_truth_labels_dir else None
        self.source_domain_name = source_domain_name
        self.target_domain_name = target_domain_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        if device is None:
            from src.utils.device import get_optimal_device
            self.device = get_optimal_device()
        else:
            self.device = device

    def evaluate_zero_shot(
        self,
        enhancer=None,
        enhancement_method: str = "raw",
        max_images: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs zero-shot inference on target-domain images,
        recording detection counts, confidence distributions, and latency.
        """
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [p for p in self.target_images_dir.glob("*") if p.suffix.lower() in img_exts]

        if max_images:
            image_files = image_files[:max_images]

        total_images = len(image_files)
        total_detections = 0
        detected_images_count = 0
        confidences = []
        species_histogram = {}
        latencies = []
        sample_results = []

        for p in image_files:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if enhancer and enhancement_method != "raw":
                enhanced_rgb, _ = enhancer.enhance(rgb, method=enhancement_method)
            else:
                enhanced_rgb = rgb

            t0 = time.perf_counter()
            results = self.model.predict(
                enhanced_rgb,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)

            res = results[0]
            boxes = res.boxes
            num_boxes = len(boxes) if boxes is not None else 0

            if num_boxes > 0:
                detected_images_count += 1
                total_detections += num_boxes

                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    species = res.names.get(cls_id, f"Class_{cls_id}")
                    confidences.append(conf)
                    species_histogram[species] = species_histogram.get(species, 0) + 1

            if len(sample_results) < 5:
                sample_results.append({
                    "image": p.name,
                    "num_detections": num_boxes,
                    "detections": [
                        {"species": res.names.get(int(b.cls[0]), "Fish"), "confidence": round(float(b.conf[0]), 3)}
                        for b in (boxes if boxes is not None else [])
                    ]
                })

        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return {
            "source_domain": self.source_domain_name,
            "target_domain": self.target_domain_name,
            "enhancement_applied": enhancement_method,
            "total_images_evaluated": total_images,
            "images_with_detections": detected_images_count,
            "detection_hit_rate": round(detected_images_count / max(1, total_images), 3),
            "total_objects_detected": total_detections,
            "average_confidence": round(avg_conf, 3),
            "average_inference_latency_ms": round(avg_latency, 2),
            "detected_species_distribution": species_histogram,
            "sample_detections": sample_results
        }
