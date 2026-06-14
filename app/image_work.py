import io
import numpy as np
from PIL import Image, ImageSequence

import torch
import torch.nn as nn
from torchvision import transforms

def read_image(uploaded_file):
    image = Image.open(uploaded_file)

    if uploaded_file.name.lower().endswith((".tif", ".tiff")):
        try:
            image = next(ImageSequence.Iterator(image))
        except Exception:
            pass

    image = image.convert("RGB")
    return image

def preprocess_image(image):
    transform_list = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)]

    transform = transforms.Compose(transform_list)
    x = transform(image)
    
    x = x.unsqueeze(0)

    return x
  
def predict_mask(model, device, image, threshold):
    original_size = image.size

    x = preprocess_image(image).to(device)
    model.eval()

    with torch.no_grad():
        logits = model(x)
        
        if isinstance(logits, list):
            logits = logits[-1]
            
        logits = nn.functional.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).float()

    mask_np = mask.squeeze().detach().cpu().numpy()
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_img = mask_img.resize(original_size, resample=Image.NEAREST)
    final_mask_np = np.array(mask_img).astype(np.float32) / 255.0

    return final_mask_np

def make_overlay(image, mask, alpha):
    image_np = np.array(image).astype(np.float32)

    if image_np.ndim == 2:
        image_np = np.stack([image_np, image_np, image_np], axis=-1)

    if image_np.shape[-1] == 4:
        image_np = image_np[:, :, :3]

    overlay = image_np.copy()
    mask_bool = mask > 0.5

    color = np.array([255, 0, 0], dtype=np.float32)

    overlay[mask_bool] = (
        image_np[mask_bool] * (1.0 - alpha)
        + color * alpha)

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)

def make_mask_image(mask):
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    return mask_img
  
def image_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()
