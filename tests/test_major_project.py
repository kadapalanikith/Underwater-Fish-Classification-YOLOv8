"""
Test Suite for DeepSea AI Major Project
Tests: preprocessing correctness, model loading, detection schema,
invalid file handling, cross-domain evaluator, and API integration.
"""
import os
import sys
import json
import io
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.enhancement import UnderwaterEnhancer
from src.data.validator import DatasetValidator
from src.utils.device import get_optimal_device


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def enhancer():
    return UnderwaterEnhancer()


@pytest.fixture(scope="session")
def sample_image_rgb():
    """Synthetic 640×640 RGB image for testing."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def sample_pil_image():
    """Synthetic PIL Image."""
    arr = np.random.randint(0, 200, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture(scope="session")
def qualitative_sample_images():
    root = Path(__file__).resolve().parent.parent / "qualitative_samples"
    return sorted(root.glob("*.jpg"))


# ─── Enhancement Tests ────────────────────────────────────────────────────────

class TestEnhancement:
    def test_raw_passthrough(self, enhancer, sample_image_rgb):
        out, meta = enhancer.enhance(sample_image_rgb, method="raw")
        assert out.shape == sample_image_rgb.shape, "Raw mode must not change shape"
        assert meta["method"] == "raw"

    def test_clahe_shape_preserved(self, enhancer, sample_image_rgb):
        out, meta = enhancer.enhance(sample_image_rgb, method="clahe")
        assert out.shape == sample_image_rgb.shape
        assert meta["pixel_min"] >= 0 and meta["pixel_max"] <= 255

    def test_gray_world_pixel_range(self, enhancer, sample_image_rgb):
        out, meta = enhancer.enhance(sample_image_rgb, method="gray_world")
        assert out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255

    def test_fusion_shape_preserved(self, enhancer, sample_image_rgb):
        out, meta = enhancer.enhance(sample_image_rgb, method="fusion")
        assert out.shape == sample_image_rgb.shape

    def test_pil_input_accepted(self, enhancer, sample_pil_image):
        out, meta = enhancer.enhance(sample_pil_image, method="clahe")
        assert isinstance(out, np.ndarray)
        assert out.shape[2] == 3

    def test_unknown_method_falls_back_to_raw(self, enhancer, sample_image_rgb):
        out, meta = enhancer.enhance(sample_image_rgb, method="nonexistent_method")
        assert out.shape == sample_image_rgb.shape
        assert "raw" in meta["method"]

    def test_rgba_input_converted(self, enhancer):
        rgba = np.random.randint(0, 255, (320, 320, 4), dtype=np.uint8)
        out, meta = enhancer.enhance(rgba, method="clahe")
        assert out.shape[2] == 3

    def test_latency_is_positive(self, enhancer, sample_image_rgb):
        _, meta = enhancer.enhance(sample_image_rgb, method="fusion")
        assert meta["latency_ms"] >= 0

    def test_real_sample_images(self, enhancer, qualitative_sample_images):
        if not qualitative_sample_images:
            pytest.skip("No qualitative_samples images found")
        img_path = str(qualitative_sample_images[0])
        out, meta = enhancer.enhance(img_path, method="fusion")
        assert out is not None
        assert isinstance(out, np.ndarray)

    def test_invalid_path_raises(self, enhancer):
        with pytest.raises(ValueError):
            enhancer.enhance("nonexistent_path_xyz.jpg", method="raw")


# ─── Dataset Validator Tests ──────────────────────────────────────────────────

class TestDatasetValidator:
    def test_validator_instantiation(self):
        v = DatasetValidator(".")
        assert len(v.class_names) == 13

    def test_default_class_list(self):
        v = DatasetValidator(".")
        assert "AngelFish" in v.class_names
        assert "ZebraFish" in v.class_names
        assert "YellowTang" in v.class_names

    def test_polygon_rows_are_valid_instances(self, tmp_path):
        split_dir = tmp_path / "train"
        (split_dir / "images").mkdir(parents=True)
        (split_dir / "labels").mkdir()
        (split_dir / "images" / "sample.jpg").write_bytes(b"")
        (split_dir / "labels" / "sample.txt").write_text(
            "\n".join([
                "0 0.5 0.5 0.2 0.2",
                "1 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9",
                "2 0.1 0.1 0.2 0.2 0.3 0.1",
            ]),
            encoding="utf-8",
        )

        result = DatasetValidator(tmp_path, class_names=["A", "B", "C"]).validate_split("train")

        assert result["box_rows"] == 1
        assert result["polygon_rows"] == 2
        assert result["converted_polygon_boxes"] == 2
        assert result["total_boxes"] == 3
        assert result["class_counts"] == {"A": 1, "B": 1, "C": 1}
        assert result["invalid_annotations_count"] == 0
        assert result["out_of_bounds_boxes"] == 0

    def test_invalid_and_out_of_bounds_annotation_rows(self, tmp_path):
        split_dir = tmp_path / "train"
        (split_dir / "images").mkdir(parents=True)
        (split_dir / "labels").mkdir()
        (split_dir / "images" / "sample.jpg").write_bytes(b"")
        (split_dir / "labels" / "sample.txt").write_text(
            "\n".join([
                "0 0.5 0.5 0.2 0.2",
                "1 0.1 0.1 1.2 0.1 1.2 0.8 0.1 0.8",
                "2 0.5 0.5 0.5 0.5 0.5 0.5",
                "3 0.1 0.2 0.3",
                "4 0.1 0.1 0.2 0.2 0.3",
                "99 0.5 0.5 0.1 0.1",
            ]),
            encoding="utf-8",
        )

        result = DatasetValidator(
            tmp_path, class_names=["A", "B", "C", "D", "E"]
        ).validate_split("train")

        assert result["box_rows"] == 1
        assert result["polygon_rows"] == 1
        assert result["total_boxes"] == 2
        assert result["class_counts"] == {"A": 1, "B": 1, "C": 0, "D": 0, "E": 0}
        assert result["out_of_bounds_boxes"] == 1
        assert result["invalid_annotations_count"] == 4

    def test_file_hash_consistency(self, qualitative_sample_images):
        if not qualitative_sample_images:
            pytest.skip("No qualitative_samples images found")
        h1 = DatasetValidator.compute_file_hash(qualitative_sample_images[0])
        h2 = DatasetValidator.compute_file_hash(qualitative_sample_images[0])
        assert h1 == h2, "Same file must produce identical hash"

    def test_different_files_different_hashes(self, qualitative_sample_images):
        if len(qualitative_sample_images) < 2:
            pytest.skip("Need at least 2 test images")
        h1 = DatasetValidator.compute_file_hash(qualitative_sample_images[0])
        h2 = DatasetValidator.compute_file_hash(qualitative_sample_images[1])
        assert h1 != h2, "Different image files should have different hashes"


# ─── Device Selection Tests ───────────────────────────────────────────────────

class TestDevice:
    def test_device_is_string(self):
        dev = get_optimal_device()
        assert isinstance(dev, str)

    def test_device_is_valid(self):
        dev = get_optimal_device()
        assert dev in ("cpu", "cuda:0", "mps")

    def test_device_is_cached(self):
        dev1 = get_optimal_device()
        dev2 = get_optimal_device()
        assert dev1 == dev2, "Device selection must be deterministic"


# ─── Model Loading Tests ──────────────────────────────────────────────────────

class TestModelLoading:
    def test_best_pt_exists(self):
        model_path = Path(__file__).resolve().parent.parent / "models" / "best.pt"
        assert model_path.exists(), "models/best.pt checkpoint must exist"

    def test_best_pt_size(self):
        model_path = Path(__file__).resolve().parent.parent / "models" / "best.pt"
        size_mb = model_path.stat().st_size / (1024 * 1024)
        # YOLOv8n should be ~6MB
        assert 4.0 < size_mb < 15.0, f"Unexpected model size: {size_mb:.2f} MB"

    def test_yolo_loads(self):
        from ultralytics import YOLO
        model_path = str(Path(__file__).resolve().parent.parent / "models" / "best.pt")
        model = YOLO(model_path)
        assert model is not None

    def test_yolo_class_count(self):
        from ultralytics import YOLO
        model_path = str(Path(__file__).resolve().parent.parent / "models" / "best.pt")
        model = YOLO(model_path)
        assert model.model.nc == 13, f"Expected 13 classes, got {model.model.nc}"

    def test_yolo_class_names(self):
        from ultralytics import YOLO
        model_path = str(Path(__file__).resolve().parent.parent / "models" / "best.pt")
        model = YOLO(model_path)
        names = list(model.names.values())
        assert "ClownFish" in names or "Gourami" in names, "Expected fish species in model class names"


# ─── Inference Tests ─────────────────────────────────────────────────────────

class TestInference:
    @pytest.fixture(scope="class")
    def loaded_model(self):
        from ultralytics import YOLO
        model_path = str(Path(__file__).resolve().parent.parent / "models" / "best.pt")
        return YOLO(model_path)

    def test_prediction_on_dummy_image(self, loaded_model):
        device = get_optimal_device()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        results = loaded_model.predict(dummy, conf=0.25, verbose=False, device=device)
        assert results is not None
        assert len(results) > 0

    def test_result_has_boxes_attribute(self, loaded_model):
        device = get_optimal_device()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        results = loaded_model.predict(dummy, conf=0.25, verbose=False, device=device)
        res = results[0]
        assert hasattr(res, "boxes")

    def test_result_has_names(self, loaded_model):
        device = get_optimal_device()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        results = loaded_model.predict(dummy, conf=0.25, verbose=False, device=device)
        res = results[0]
        assert hasattr(res, "names")
        assert len(res.names) == 13

    def test_annotated_image_has_correct_dims(self, loaded_model):
        device = get_optimal_device()
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        results = loaded_model.predict(dummy, conf=0.25, verbose=False, device=device)
        annotated = results[0].plot()
        assert annotated.shape[0] == 480
        assert annotated.shape[1] == 640

    def test_qualitative_sample_prediction(self, loaded_model, qualitative_sample_images):
        if not qualitative_sample_images:
            pytest.skip("No test sample images available")
        device = get_optimal_device()
        import cv2
        img_bgr = cv2.imread(str(qualitative_sample_images[0]))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = loaded_model.predict(img_rgb, conf=0.25, verbose=False, device=device)
        assert results is not None
        # At least the result object should have the right structure
        res = results[0]
        assert hasattr(res, "boxes") and hasattr(res, "names")


# ─── Enhancement + Detection Integration Tests ───────────────────────────────

class TestEnhancementPipeline:
    def test_full_pipeline_raw_to_output(self, qualitative_sample_images):
        if not qualitative_sample_images:
            pytest.skip("No qualitative_samples images")
        from ultralytics import YOLO
        device = get_optimal_device()
        model = YOLO(str(Path(__file__).resolve().parent.parent / "models" / "best.pt"))
        enh = UnderwaterEnhancer()
        enhanced_rgb, _ = enh.enhance(str(qualitative_sample_images[0]), method="fusion")
        results = model.predict(enhanced_rgb, conf=0.25, verbose=False, device=device)
        assert results is not None

    def test_all_methods_return_equal_shape(self, enhancer, sample_image_rgb):
        shapes = set()
        for method in ["raw", "clahe", "gray_world", "fusion"]:
            out, _ = enhancer.enhance(sample_image_rgb, method=method)
            shapes.add(out.shape)
        assert len(shapes) == 1, "All methods must preserve input dimensions"


# ─── Output Files Tests ───────────────────────────────────────────────────────

class TestOutputFiles:
    def test_enhancement_benchmark_exists(self):
        p = Path(__file__).resolve().parent.parent / "outputs" / "enhancement_benchmark.json"
        assert p.exists(), "Enhancement benchmark output file must exist after running the experiment"

    def test_enhancement_benchmark_schema(self):
        p = Path(__file__).resolve().parent.parent / "outputs" / "enhancement_benchmark.json"
        if not p.exists():
            pytest.skip("Benchmark file not yet generated")
        with open(p) as f:
            data = json.load(f)
        assert "raw" in data
        assert "total_detections" in data["raw"]
        assert "avg_confidence" in data["raw"]

    def test_model_benchmark_exists(self):
        p = Path(__file__).resolve().parent.parent / "outputs" / "model_benchmark_comparison.json"
        assert p.exists(), "Model benchmark comparison file must exist"

    def test_model_benchmark_schema(self):
        p = Path(__file__).resolve().parent.parent / "outputs" / "model_benchmark_comparison.json"
        if not p.exists():
            pytest.skip("Model benchmark not yet generated")
        with open(p) as f:
            data = json.load(f)
        assert "models" in data
        assert len(data["models"]) >= 1
