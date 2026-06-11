import io
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import urllib.request
import rasterio
from rasterio.io import MemoryFile

import streamlit as st

from PIL import Image, ImageSequence
import cv2
import torch
from torchvision import transforms

from model import build_model

WEIGHTS_PATH = "best_weights.pth"
WEIGHTS_URL = "https://github.com/Fl1nixxx/DLS-project/releases/download/v1.2/best_weights.pth"

IMAGE_SIZE = 352
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

def download_weights():
    if os.path.exists(WEIGHTS_PATH):
        return
      
    with st.spinner("Скачиваю веса модели из GitHub Release..."):
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)

    st.success("Веса модели скачаны.")

@st.cache_resource
def load_segmentation_model():
    download_weights()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(in_channels=3, out_channels=1).to(device)
    
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def define_pixel_area(uploaded_file):
    uploaded_file.seek(0)
    with MemoryFile(uploaded_file.read()) as memfile:
        with memfile.open() as src:
            crs = src.crs
            x, y = src.res
            if crs and crs.is_projected:
                pixel_area = x * y
            else:
                lat = src.bounds.bottom + (src.bounds.top - src.bounds.bottom) / 2
                meters_per_degree_lat = 111132
                meters_per_degree_lon = meters_per_degree_lat * math.cos(math.radians(lat))
                pixel_area = (x * meters_per_degree_lon) * (y * meters_per_degree_lat)
    return pixel_area

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

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).float()

    mask_np = mask.squeeze().detach().cpu().numpy()

    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_img = mask_img.resize(original_size, resample=Image.NEAREST)

    mask_np = np.array(mask_img).astype(np.float32) / 255.0

    return mask_np

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
        + color * alpha
    )

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

def count_building_area(image, mask, pixel_area, noise_threshold=7):

    FONT_SIZE = 3.0
    THICKNESS = 8
    PADDING = 25

    TEXT_COLOR = (255, 255, 255)
    BG_COLOR = (0, 0, 0)

    result_img = np.array(image).copy()

    binary_mask = np.array(mask)
    if binary_mask.max() == 1:
        binary_mask = (binary_mask * 255).astype(np.uint8)
    else:
        binary_mask = (binary_mask > 127).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    building_idx = 0
    buildings_report = []

    for cnt in contours:
        pixel_count = cv2.contourArea(cnt)
        area_sqm = pixel_count * pixel_area

        if area_sqm < noise_threshold:
            continue

        building_idx += 1
        area_rounded = round(area_sqm, 2)
        buildings_report.append((building_idx, area_rounded))

        cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 3)

        x, y, w, h = cv2.boundingRect(cnt)

        text = f"ID {building_idx}: {area_rounded} m2"

        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, THICKNESS)

        text_x = int(x + w / 2) - int(text_width / 2)
        text_y = int(y + h / 2) + int(text_height / 2)

        box_x1 = text_x - PADDING
        box_y1 = text_y - text_height - PADDING
        box_x2 = text_x + text_width + PADDING
        box_y2 = text_y + baseline + PADDING

        cv2.rectangle(result_img, (box_x1, box_y1), (box_x2, box_y2), BG_COLOR, -1)

        cv2.rectangle(result_img, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 4)

        cv2.putText(result_img,text,(text_x, text_y),cv2.FONT_HERSHEY_SIMPLEX,FONT_SIZE,TEXT_COLOR,THICKNESS,cv2.LINE_AA)
    
    return result_img




def main():
    st.set_page_config(
        page_title="Image Segmentation App",
        layout="wide")

    st.title("Image Segmentation App")
    st.write(
        "Загрузи изображение в формате `.tif`, `.tiff`"
        "Приложение построит сегментационную маску и наложит её поверх изображения.")

    with st.sidebar:
        st.header("Настройки")

        threshold = st.slider("Порог сегментации",min_value=0.0,max_value=1.0,value=0.5,step=0.05,)

        alpha = st.slider("Прозрачность маски",min_value=0.0,max_value=1.0,value=0.45,step=0.05,)

        st.divider()

        st.write("Размер входа модели:")
        st.code(f"{IMAGE_SIZE} x {IMAGE_SIZE}")

        st.write("Файл весов:")
        st.code(WEIGHTS_PATH)

        st.write("Поддерживаемые форматы:")
        st.code(".tif, .tiff")

    st.subheader("Загрузка изображения")

    uploaded_file = st.file_uploader(
        "Выбери изображение",
        type=["tif", "tiff"],)

    if uploaded_file is None:
        st.info("Загрузи изображение в формате `.tif`, `.tiff`")
        return

    try:
        image = read_image(uploaded_file)
    except Exception as e:
        st.error("Не получилось открыть изображение.")
        st.exception(e)
        return

    st.success(
        f"Файл загружен: `{uploaded_file.name}`. "
        f"Размер изображения: {image.size[0]} x {image.size[1]}")

    try:
        model, device = load_segmentation_model()
    except Exception as e:
        st.error("Ошибка при загрузке модели или весов.")
        st.exception(e)
        return

    if st.button("Запустить сегментацию"):
        with st.spinner("Модель строит маску..."):
            try:
                mask = predict_mask(model=model,device=device,image=image,threshold=threshold,)

                mask_image = make_mask_image(mask)

                overlay = make_overlay(image=image,mask=mask,alpha=alpha,)

            except Exception as e:
                st.error("Ошибка во время предсказания.")
                st.exception(e)
                return

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Mask")
            st.image(mask_image, use_container_width=True)

        with col3:
            st.subheader("Overlay")
            st.image(overlay, use_container_width=True)

        st.download_button(label="Скачать overlay PNG",data=image_to_png_bytes(overlay),file_name="overlay.png",mime="image/png",)

        st.download_button(label="Скачать mask PNG",data=image_to_png_bytes(mask_image),file_name="mask.png",mime="image/png",)
        
        try:
            pixel_area = define_pixel_area(uploaded_file)
            area_map = count_building_area(image, mask, pixel_area, noise_threshold=7)

            st.subheader("Building area")
            st.image(area_map, use_container_width=True)
        except Exception as e:
            st.warning("Не удалось рассчитать гео-координаты. Возможно, файл не содержит метаданных (не GeoTIFF).")
            st.exception(e)
            return

if __name__ == "__main__":
    main()
