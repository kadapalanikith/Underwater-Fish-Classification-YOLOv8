import os
import sys
import json
from ultralytics import YOLO
from src.preprocessing.enhancement import UnderwaterEnhancer
from src.evaluation.cross_domain import CrossDomainEvaluator
from src.utils.device import get_optimal_device

device = get_optimal_device()
model = YOLO("models/best.pt")
enhancer = UnderwaterEnhancer()

evaluator = CrossDomainEvaluator(
    model=model,
    target_images_dir="qualitative_samples",
    source_domain_name="Kaggle/Roboflow Curated Tank & Aquarium Domain",
    target_domain_name="Challenging Wild Underwater Imagery (Green/Blue Attenuation & Corals)"
)

methods = ["raw", "clahe", "gray_world", "fusion"]
domain_results = {}

for m in methods:
    print(f"Evaluating domain shift with enhancement method: {m}...")
    res = evaluator.evaluate_zero_shot(enhancer=enhancer, enhancement_method=m)
    domain_results[m] = res

print("\n--- Cross-Domain Generalization Results ---")
print(json.dumps(domain_results, indent=2))

with open("outputs/cross_domain_evaluation.json", "w") as f:
    json.dump(domain_results, f, indent=2)
print("Saved to outputs/cross_domain_evaluation.json")
