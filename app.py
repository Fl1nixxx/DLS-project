import io
import os
import urllib.request

import numpy as np
import streamlit as st
import torch

from PIL import Image, ImageSequence
from torchvision import transforms

from model import build_model

#Settings
WEIGHTS_PATH = "best_model.pth"
WEIGHTS_URL = ("https://github.com/Fl1nixxx/DLS-project/releases/download/v1.1/best_weigths.pth")

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(in_channels=3, out_channels=1)

    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key.replace("module.", "")
        clean_state_dict[clean_key] = value

    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()

    return model, device


def read_tiff(uploaded_file):
    image = Image.open(uploaded_file)

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
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ]

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

def main():
    st.set_page_config(
        page_title="TIFF Segmentation App",
        layout="wide"
    )

    st.title("TIFF Segmentation App")
    st.write(
        "Загрузи `.tif` или `.tiff`, приложение построит сегментационную маску "
        "и наложит её поверх изображения")

    with st.sidebar:
        st.header("Настройки")

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

        st.write("Размер входа модели:")
        st.code(f"{IMAGE_SIZE} x {IMAGE_SIZE}")

        st.write("Файл весов:")
        st.code(WEIGHTS_PATH)

    uploaded_file = st.file_uploader(
        "Загрузи TIFF-файл",
        type=["tif", "tiff"],
    )

    if uploaded_file is None:
        st.info("Загрузи изображение в формате `.tif` или `.tiff`.")
        return

    try:
        model, device = load_segmentation_model()
    except Exception as e:
        st.error("Ошибка при загрузке модели или весов.")
        st.exception(e)
        return

    try:
        image = read_tiff(uploaded_file)
    except Exception as e:
        st.error("Не получилось открыть TIFF-файл.")
        st.exception(e)
        return

    st.success(
        f"Файл загружен. Размер изображения: "
        f"{image.size[0]} x {image.size[1]}"
    )

    if st.button("Запустить сегментацию"):
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

        st.download_button(
            label="Скачать overlay PNG",
            data=image_to_png_bytes(overlay),
            file_name="overlay.png",
            mime="image/png",
        )

        st.download_button(
            label="Скачать mask PNG",
            data=image_to_png_bytes(mask_image),
            file_name="mask.png",
            mime="image/png",
        )


if __name__ == "__main__":
    main()
