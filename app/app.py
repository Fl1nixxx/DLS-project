import io
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import math

import urllib.request
import rasterio
from rasterio.io import MemoryFile

import streamlit as st

from PIL import Image, ImageSequence
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

from model import build_model

WEIGHTS_PATH = "best_weights.pth"
WEIGHTS_URL = "https://github.com/Fl1nixxx/DLS-project/releases/download/v1.5/best_weights.pth"

IMAGE_SIZE = 512
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
    model = build_model(1, True).to(device)
    
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def define_pixel_area(uploaded_file):
    try:
        uploaded_file.seek(0)
        with MemoryFile(uploaded_file.read()) as memfile:
            with memfile.open() as src:
                crs = src.crs
                x, y = src.res
                
                if src.bounds.left == 0.0 and src.bounds.bottom == 0.0:
                    return None 
                
                if crs and crs.is_projected:
                    pixel_area = x * y
                else:
                    lat = src.bounds.bottom + (src.bounds.top - src.bounds.bottom) / 2
                    meters_per_degree_lat = 111132
                    meters_per_degree_lon = meters_per_degree_lat * math.cos(math.radians(lat))
                    pixel_area = (x * meters_per_degree_lon) * (y * meters_per_degree_lat)
                    
        return pixel_area
    except Exception:
        return None

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
        + color * alpha
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)

def make_mask_image(mask):
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    return mask_img

def count_building_area(mask, pixel_area, noise_threshold=10):
    binary_mask = np.array(mask)
    if binary_mask.max() == 1:
        binary_mask = (binary_mask * 255).astype(np.uint8)
    else:
        binary_mask = (binary_mask > 127).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    buildings_report = []
    noise_count = 0

    for cnt in contours:
        pixel_count = cv2.contourArea(cnt)
        area_sqm = pixel_count * pixel_area

        if area_sqm < noise_threshold:
            noise_count += 1
            continue

        area_rounded = round(area_sqm, 2)
        buildings_report.append(area_rounded)
        
    total_area = sum(buildings_report)
    buildings_count = len(buildings_report)
    
    return total_area, buildings_count, noise_count

def image_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def main():
    st.set_page_config(page_title="Image Segmentation App", layout="wide")

    st.title("🏢 Image Segmentation App")
    st.write(
        "Загрузите изображение в формате `.tif`, `.tiff`, `.png` или `.jpg`. "
        "Приложение построит сегментационную маску, наложит её поверх изображения и рассчитает статистику застройки."
    )
    
    with st.sidebar:
        st.header("⚙️ Настройки")

        threshold = st.slider(
            "Порог сегментации",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )

        alpha = st.slider(
            "Прозрачность маски",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
        )

        st.divider()

        st.subheader("📐 Размер пикселя")
        pixel_mode = st.radio(
            "Способ определения:",
            ["Авто (из метаданных GeoTIFF)", "Вручную (в см² на пиксель)"]
        )

        pixel_cm2 = st.number_input(
            "Площадь пикселя (см² / px):",
            min_value=0.1,
            value=900.0,
            step=10.0,
            help="Укажите, какую площадь земной поверхности покрывает один пиксель в квадратных сантиметрах."
        )
        
        manual_pixel_area = pixel_cm2 / 10000.0

        st.divider()
        st.write("Поддерживаемые форматы:")
        st.code(".tif, .tiff, .png, .jpg, .jpeg")

    st.subheader("📂 Загрузка изображения")

    uploaded_file = st.file_uploader(
        "Выбери изображение",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file is None:
        st.info("Пожалуйста, загрузите изображение для начала работы.")
        return

    try:
        image = read_image(uploaded_file)
    except Exception as e:
        st.error("Не получилось открыть изображение.")
        st.exception(e)
        return

    st.success(
        f"Файл загружен: `{uploaded_file.name}`. "
        f"Размер изображения: {image.size[0]} x {image.size[1]} px"
    )

    try:
        model, device = load_segmentation_model()
    except Exception as e:
        st.error("Ошибка при загрузке модели или весов.")
        st.exception(e)
        return

    if "seg_results" not in st.session_state:
        st.session_state.seg_results = None

    if st.button("🚀 Запустить сегментацию", type="primary", use_container_width=True):
        with st.spinner("Модель строит маску..."):
            try:
                mask = predict_mask(
                    model=model,
                    device=device,
                    image=image,
                    threshold=threshold,
                )
                mask_image = make_mask_image(mask)
                overlay = make_overlay(
                    image=image,
                    mask=mask,
                    alpha=alpha,
                )

                st.session_state.seg_results = {
                    "raw_mask": mask,
                    "mask_image": mask_image,
                    "overlay": overlay
                }

            except Exception as e:
                st.error("Ошибка во время обработки моделью.")
                st.exception(e)
                return


    if st.session_state.seg_results is not None:
        res = st.session_state.seg_results

        try:
            if pixel_mode == "Вручную (в см² на пиксель)":
                pixel_area = manual_pixel_area
                used_manual_fallback = False
            else:
                pixel_area = define_pixel_area(uploaded_file)
                if pixel_area is None:
                    pixel_area = manual_pixel_area
                    used_manual_fallback = True
                else:
                    used_manual_fallback = False

            total_area, buildings_count, noise_count = count_building_area(
                mask=res["raw_mask"], 
                pixel_area=pixel_area, 
                noise_threshold=10
            )

        except Exception as e:
            st.error("Ошибка при динамическом расчете площади застройки.")
            st.exception(e)
            return

        if used_manual_fallback:
            st.warning(
                f"⚠️ Внимание: Файл не содержит гео-метаданных (PNG/JPG). "
                f"Использовано значение ручного ввода из настроек сайдбара ({pixel_cm2} см²/px)."
            )

        tab1, tab2, tab3 = st.tabs(["🖼️ Совмещение (Overlay)", "🎭 Маска (Mask)", "📷 Оригинал"])
        
        with tab1:
            st.image(res["overlay"].convert("RGB"), use_container_width=True)
        with tab2:
            st.image(res["mask_image"].convert("RGB"), use_container_width=True)
        with tab3:
            st.image(image.convert("RGB"), use_container_width=True)

        down_col1, down_col2 = st.columns(2)
        with down_col1:
            st.download_button(
                label="⬇️ Скачать overlay PNG",
                data=image_to_png_bytes(res["overlay"]),
                file_name=f"overlay_{uploaded_file.name}.png",
                mime="image/png",
                use_container_width=True
            )
        with down_col2:
            st.download_button(
                label="⬇️ Скачать mask PNG",
                data=image_to_png_bytes(res["mask_image"]),
                file_name=f"mask_{uploaded_file.name}.png",
                mime="image/png",
                use_container_width=True
            )

        st.markdown("---")
        st.subheader("📊 Статистика застройки")

        col_stat1, col_stat2, col_stat3 = st.columns(3)

        with col_stat1:
            st.metric(
                label="📐 Суммарная площадь застройки", 
                value=f"{total_area:,.2f} м²".replace(",", " ")
            )

        with col_stat2:
            st.metric(
                label="🏠 Количество зданий", 
                value=f"{buildings_count} шт."
            )

        with col_stat3:
            st.metric(
                label="🗑️ Удалено мелкого шума (<10м²)", 
                value=f"{noise_count} объектов"
            )


if __name__ == "__main__":
    main()

