from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"

# Canonical deployment checkpoints exported by the updated notebook.
SEGMENTATION_MODEL_PATH = CHECKPOINT_DIR / "unetplusplus_final_research.pth"
CLASSIFIER_MODEL_PATH = CHECKPOINT_DIR / "resnet18_final_research.pth"

# Optional legacy/training checkpoints retained for reproducibility.
LEGACY_SEGMENTATION_MODEL_PATH = CHECKPOINT_DIR / "unetplusplus_best_validation.pth"
LEGACY_CLASSIFIER_MODEL_PATH = CHECKPOINT_DIR / "resnet18_best_validation.pth"

MODEL_NAME = "U-Net++"
CLASSIFIER_NAME = "ResNet18"
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = None
IN_CHANNELS = 3
NUM_CLASSES = 1
IMAGE_SIZE = 128
THRESHOLD = 0.5
TUMOR_DETECTION_THRESHOLD = 0.1
GRADCAM_THRESHOLD = 0.5
GRADCAM_SAMPLE_LIMIT = 20
USE_AMP = False
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff"}
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/tiff"}
MAX_UPLOAD_MB = 20
MAX_IMAGE_DIMENSION = 4096
OVERLAY_ALPHA = 0.45
TUMOR_COLOR = (239, 68, 68)

APP_TITLE = "🧠 NeuroScan AI"
APP_SUBTITLE = "Brain Tumor Segmentation + Classification + Explainability"
APP_DESCRIPTION = (
    "BraTS 2020 brain MRI analysis using CLAHE, Gaussian filtering, "
    "Gabor and Sobel feature channels, U-Net++ segmentation, ResNet18 "
    "binary classification and Grad-CAM."
)
APP_VERSION = "2.1.0"
PAGE_ICON = "🧠"
LAYOUT = "wide"
PRIMARY_COLOR = "#2563EB"
SUCCESS_COLOR = "#16A34A"
WARNING_COLOR = "#F59E0B"
ERROR_COLOR = "#DC2626"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
SEED = 42
