from pathlib import Path
import logging
import streamlit as st
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision import models
import config
logger = logging.getLogger(__name__)
if config.DEVICE.type == "cuda": torch.backends.cudnn.benchmark = True

def _load_state_dict(path: Path):
    checkpoint = torch.load(path, map_location=config.DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint

def build_segmentation_model():
    return smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)

def build_classifier():
    classifier = models.resnet18(weights=None)
    classifier.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    classifier.fc = nn.Linear(classifier.fc.in_features, 1)
    return classifier

@st.cache_resource(show_spinner=False)
def load_segmentation_model():
    model = build_segmentation_model(); device = config.DEVICE
    path = config.SEGMENTATION_MODEL_PATH
    if not path.exists():
        path = config.LEGACY_SEGMENTATION_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Segmentation checkpoint not found. Expected {config.SEGMENTATION_MODEL_PATH} "
            f"or {config.LEGACY_SEGMENTATION_MODEL_PATH}."
        )
    model.load_state_dict(_load_state_dict(path)); model.to(device).eval()
    for p in model.parameters(): p.requires_grad = False
    return model, device, path

@st.cache_resource(show_spinner=False)
def load_classifier():
    path = config.CLASSIFIER_MODEL_PATH
    if not path.exists():
        path = config.LEGACY_CLASSIFIER_MODEL_PATH
    if not path.exists():
        return None, config.DEVICE, None
    classifier = build_classifier()
    classifier.load_state_dict(_load_state_dict(path))
    classifier.to(config.DEVICE).eval()
    return classifier, config.DEVICE, path
