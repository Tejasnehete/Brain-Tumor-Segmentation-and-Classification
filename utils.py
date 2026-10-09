from __future__ import annotations
import io, logging, time
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
import config
logger = logging.getLogger(__name__)

def validate_image(file_bytes: bytes, filename: str) -> Image.Image:
    extension = Path(filename).suffix.lower().replace(".", "")
    if extension not in config.ALLOWED_EXTENSIONS: raise ValueError(f"Unsupported file type: {extension}")
    if len(file_bytes) > config.MAX_UPLOAD_MB * 1024 * 1024: raise ValueError(f"Maximum upload size is {config.MAX_UPLOAD_MB} MB.")
    try:
        probe = Image.open(io.BytesIO(file_bytes)); probe.verify()
    except UnidentifiedImageError: raise ValueError("Uploaded file is not a valid image.")
    except Exception as e: raise ValueError(str(e))
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(file_bytes))).convert("RGB")
    if max(image.size) > config.MAX_IMAGE_DIMENSION: raise ValueError("Image dimensions exceed the maximum allowed.")
    return image

def convert_to_grayscale(image):
    """Convert an RGB/PIL image or array to a single-channel grayscale MRI image."""
    if isinstance(image, Image.Image):
        return np.array(image.convert("L"))
    arr = np.asarray(image)
    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr.astype(np.uint8)

def clahe_equalization(image): return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
def gaussian_noise_removal(image): return cv2.GaussianBlur(image, (5, 5), sigmaX=0)
def normalize_image(image):
    image = image.astype(np.float32); image -= image.min(); m=image.max()
    if m > 0: image /= m
    return image

def create_gabor_kernels():
    return [cv2.getGaborKernel((21,21), 5.0, theta, 10.0, 0.5, 0, cv2.CV_32F) for theta in [0,np.pi/4,np.pi/2,3*np.pi/4]]
GABOR_KERNELS=create_gabor_kernels()

def apply_gabor_filter(image):
    responses=[cv2.filter2D(image,cv2.CV_32F,k) for k in GABOR_KERNELS]
    gabor=np.max(np.stack(responses),axis=0); gabor=cv2.normalize(gabor,None,0,1,cv2.NORM_MINMAX)
    return gabor.astype(np.float32)

def preprocess_mri_array(image): return normalize_image(gaussian_noise_removal(clahe_equalization(image)))

def apply_sobel_filter(image):
    """Compute normalized Sobel gradient magnitude."""
    sx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sx ** 2 + sy ** 2)
    return cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

def get_preprocessing_steps(image):
    """Return every preprocessing stage used by the inference pipeline.

    The numerical operations and parameters intentionally match preprocess().
    """
    grayscale = convert_to_grayscale(image)
    clahe = clahe_equalization(grayscale)
    gaussian = gaussian_noise_removal(clahe)
    normalized = normalize_image(gaussian).astype(np.float32)
    gabor = apply_gabor_filter(normalized)
    sobel = apply_sobel_filter(normalized)
    channels = np.stack([normalized, gabor, sobel]).astype(np.float32)
    resized = np.stack([
        cv2.resize(ch, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        for ch in channels
    ]).astype(np.float32)
    return {
        "original": np.array(image.convert("L")),
        "grayscale": grayscale,
        "clahe": clahe,
        "gaussian": gaussian,
        "normalized": normalized,
        "gabor": gabor,
        "sobel": sobel,
        "three_channel": channels,
        "resized_channels": resized,
    }

def create_three_channel_image_from_gray(gray):
    mri=preprocess_mri_array(gray); gabor=apply_gabor_filter(mri)
    sx=cv2.Sobel(mri,cv2.CV_32F,1,0,ksize=3); sy=cv2.Sobel(mri,cv2.CV_32F,0,1,ksize=3)
    sobel=cv2.normalize(np.sqrt(sx**2+sy**2),None,0,1,cv2.NORM_MINMAX)
    return np.stack([mri,gabor,sobel]).astype(np.float32)

def create_three_channel_image(image): return create_three_channel_image_from_gray(np.array(image.convert("L")))

def preprocess(image):
    channels=create_three_channel_image(image)
    resized=[cv2.resize(ch,(config.IMAGE_SIZE,config.IMAGE_SIZE),interpolation=cv2.INTER_LINEAR) for ch in channels]
    return torch.tensor(np.stack(resized),dtype=torch.float32).unsqueeze(0)

@torch.no_grad()
def run_inference(model,device,image_tensor):
    image_tensor=image_tensor.to(device)
    if device.type=="cuda": torch.cuda.synchronize()
    start=time.perf_counter()
    if config.USE_AMP and device.type=="cuda":
        with torch.amp.autocast(device_type="cuda",enabled=True): logits=model(image_tensor)
    else: logits=model(image_tensor)
    if isinstance(logits,(tuple,list)): logits=logits[-1]
    probability=torch.sigmoid(logits)
    if device.type=="cuda": torch.cuda.synchronize()
    return probability.squeeze().cpu().numpy(), time.perf_counter()-start

def clean_prediction(binary_mask):
    k=np.ones((3,3),np.uint8); binary_mask=cv2.morphologyEx(binary_mask,cv2.MORPH_OPEN,k)
    return cv2.morphologyEx(binary_mask,cv2.MORPH_CLOSE,k)

def keep_largest_component(binary_mask):
    number,labels,stats,_=cv2.connectedComponentsWithStats(binary_mask,connectivity=8)
    if number<=1:return binary_mask
    largest=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]); out=np.zeros_like(binary_mask); out[labels==largest]=255; return out

def generate_mask(probability_map,original_size,threshold=config.THRESHOLD):
    binary=(probability_map>=threshold).astype(np.uint8)*255; binary=keep_largest_component(clean_prediction(binary))
    return Image.fromarray(binary,mode="L").resize(original_size,Image.NEAREST)

def create_overlay(original_image,mask):
    original=np.array(original_image.convert("RGB")); overlay=original.copy(); selected=np.array(mask)>0; color=np.array(config.TUMOR_COLOR,dtype=np.uint8)
    overlay[selected]=(config.OVERLAY_ALPHA*color+(1-config.OVERLAY_ALPHA)*overlay[selected]).astype(np.uint8); return Image.fromarray(overlay)

def draw_contours(overlay,mask):
    a=np.array(overlay); contours,_=cv2.findContours(np.array(mask),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(a,contours,-1,(0,255,0),2); return Image.fromarray(a)

def tumor_statistics(mask):
    a=np.array(mask); pixels=int(np.count_nonzero(a)); total=int(a.size); return {"tumor_pixels":pixels,"total_pixels":total,"coverage_percent":round(pixels/total*100,2)}

def predict_classification(classifier,image_tensor,device):
    classifier.eval(); image_tensor=image_tensor.unsqueeze(0) if image_tensor.dim()==3 else image_tensor; image_tensor=image_tensor.to(device)
    with torch.no_grad(): p=torch.sigmoid(classifier(image_tensor).view(-1)).item()
    return int(p>=.5),np.array([1-p,p],dtype=np.float32)

def generate_gradcam(classifier,image_tensor,device):
    from pytorch_grad_cam import GradCAM
    classifier.eval(); x=image_tensor.unsqueeze(0).to(device) if image_tensor.dim()==3 else image_tensor.to(device)
    cam=GradCAM(model=classifier,target_layers=[classifier.layer4[-1]]); grayscale_cam=cam(input_tensor=x,targets=None)[0]
    with torch.no_grad(): p=torch.sigmoid(classifier(x).view(-1)).item()
    return grayscale_cam,int(p>=.5),p

def make_gradcam_overlay(mri,cam):
    from pytorch_grad_cam.utils.image import show_cam_on_image
    mri=np.clip(mri,0,1).astype(np.float32); return show_cam_on_image(np.stack([mri,mri,mri],axis=-1),cam,use_rgb=True)

def calculate_gradcam_overlap(cam,mask,threshold=config.GRADCAM_THRESHOLD):
    cb=cam>=threshold; mb=mask>=.5; inter=np.logical_and(cb,mb).sum(); union=np.logical_or(cb,mb).sum(); return float(inter/(union+1e-8)),float(inter/mb.sum()) if mb.sum() else 0.0

def image_to_bytes(image,format="PNG"):
    b=io.BytesIO(); image.save(b,format=format); return b.getvalue()

def predict_image(model,device,image):
    tensor=preprocess(image); prob,t=run_inference(model,device,tensor); mask=generate_mask(prob,image.size); overlay=draw_contours(create_overlay(image,mask),mask)
    return {"input_tensor":tensor,"mask":mask,"overlay":overlay,"probability_map":prob,"statistics":tumor_statistics(mask),"time":t}
