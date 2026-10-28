"""Aplikasi Streamlit untuk deteksi penyakit kulit dari gambar menggunakan model .pt."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "model.pt"
DEFAULT_LABELS_PATH = BASE_DIR / "models" / "labels.txt"

NORMALIZATIONS = {
    "ImageNet (mean/std)": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "Skala 0-1 (tanpa normalisasi)": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    "Skala -1 sampai 1": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
}

st.set_page_config(page_title="Deteksi Penyakit Kulit", page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Memuat model...")
def load_model(path: str, _mtime: float):
    """Muat model .pt. Mengembalikan (model, jenis).

    Urutan percobaan: TorchScript -> Ultralytics YOLO -> model PyTorch utuh.
    `_mtime` hanya dipakai agar cache diperbarui saat file berubah.
    """
    errors = []

    try:
        model = torch.jit.load(path, map_location="cpu")
        model.eval()
        return model, "torchscript"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"TorchScript: {exc}")

    try:
        from ultralytics import YOLO

        return YOLO(path), "yolo"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Ultralytics: {exc}")

    try:
        # weights_only=False diperlukan untuk model utuh hasil torch.save(model).
        # Hanya muat file model yang Anda percayai.
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.nn.Module):
            obj.eval()
            return obj, "pytorch"
        if isinstance(obj, dict):
            errors.append(
                "File berisi state_dict/checkpoint tanpa arsitektur. "
                "Simpan model sebagai TorchScript (torch.jit.script/trace) "
                "atau torch.save(model)."
            )
        else:
            errors.append(f"Tipe objek tidak didukung: {type(obj).__name__}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"torch.load: {exc}")

    raise RuntimeError("Gagal memuat model.\n- " + "\n- ".join(errors))


def preprocess(image: Image.Image, size: int, mean, std) -> torch.Tensor:
    img = image.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H, W, C
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # C, H, W
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return ((tensor - mean_t) / std_t).unsqueeze(0)


def to_probabilities(output) -> np.ndarray:
    if isinstance(output, (list, tuple)):
        output = output[0]
    out = output.detach().float().cpu().reshape(-1)
    if out.numel() == 1:  # keluaran sigmoid biner
        p = out.item()
        p = p if 0.0 <= p <= 1.0 else float(torch.sigmoid(out).item())
        return np.array([1.0 - p, p])
    is_prob = bool((out >= 0).all() and (out <= 1).all() and abs(out.sum().item() - 1.0) < 1e-3)
    return (out if is_prob else torch.softmax(out, dim=0)).numpy()


def predict_torch(model, image: Image.Image, size: int, mean, std) -> np.ndarray:
    x = preprocess(image, size, mean, std)
    with torch.inference_mode():
        return to_probabilities(model(x))


def predict_yolo(model, image: Image.Image, size: int):
    """Mengembalikan (probs | None, nama_kelas, gambar_anotasi | None, daftar_deteksi)."""
    result = model.predict(image, imgsz=size, verbose=False)[0]
    names = [result.names[i] for i in sorted(result.names)]
    if getattr(result, "probs", None) is not None:
        return result.probs.data.cpu().numpy(), names, None, []
    annotated = Image.fromarray(result.plot()[..., ::-1])  # BGR -> RGB
    detections = [
        {
            "Kelas": result.names[int(b.cls)],
            "Keyakinan": float(b.conf),
        }
        for b in result.boxes
    ]
    return None, names, annotated, detections


# --------------------------------------------------------------------------- #
# Helper UI
# --------------------------------------------------------------------------- #
def parse_labels(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def resolve_model_path(uploaded) -> Path | None:
    if uploaded is not None:
        data = uploaded.getvalue()
        digest = hashlib.sha256(data).hexdigest()[:16]
        path = Path(tempfile.gettempdir()) / f"skin_model_{digest}.pt"
        if not path.exists():
            path.write_bytes(data)
        return path
    return DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else None


def show_ranking(probs: np.ndarray, labels: list[str], top_k: int) -> None:
    names = [labels[i] if i < len(labels) else f"Kelas {i}" for i in range(len(probs))]
    order = np.argsort(probs)[::-1][:top_k]
    top = order[0]

    st.metric("Prediksi utama", names[top], f"{probs[top] * 100:.1f}% yakin")
    st.progress(float(probs[top]))

    df = pd.DataFrame(
        {"Penyakit": [names[i] for i in order], "Probabilitas (%)": [probs[i] * 100 for i in order]}
    )
    st.bar_chart(df, x="Penyakit", y="Probabilitas (%)", horizontal=True)
    st.dataframe(
        df.style.format({"Probabilitas (%)": "{:.2f}"}),
        hide_index=True,
        use_container_width=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Pengaturan")

    uploaded_model = st.file_uploader(
        "Unggah model (.pt)",
        type=["pt"],
        help=f"Kosongkan untuk memakai {DEFAULT_MODEL_PATH.relative_to(BASE_DIR)}",
    )

    default_labels = (
        DEFAULT_LABELS_PATH.read_text(encoding="utf-8") if DEFAULT_LABELS_PATH.exists() else ""
    )
    labels_text = st.text_area(
        "Nama kelas (satu per baris, sesuai urutan indeks model)",
        value=default_labels,
        height=160,
        help="Bisa juga disimpan di models/labels.txt. Model YOLO memakai nama kelas bawaannya jika kosong.",
    )

    img_size = st.number_input("Ukuran input gambar (px)", 32, 1024, 224, step=32)
    norm_name = st.selectbox("Normalisasi", list(NORMALIZATIONS), help="Samakan dengan preprocessing saat training.")
    top_k = st.slider("Jumlah hasil teratas", 1, 20, 5)

mean, std = NORMALIZATIONS[norm_name]
user_labels = parse_labels(labels_text)

# --------------------------------------------------------------------------- #
# Halaman utama
# --------------------------------------------------------------------------- #
st.title("🩺 Deteksi Penyakit Kulit")
st.caption("Unggah foto area kulit yang ingin diperiksa, lalu model akan memberikan perkiraan jenis penyakitnya.")
st.warning(
    "Hasil ini hanya alat bantu dan **bukan diagnosis medis**. "
    "Konsultasikan dengan dokter atau dokter spesialis kulit untuk pemeriksaan yang akurat.",
    icon="⚠️",
)

model_path = resolve_model_path(uploaded_model)
if model_path is None:
    st.info(
        f"Belum ada model. Unggah file `.pt` di sidebar atau letakkan di `{DEFAULT_MODEL_PATH.relative_to(BASE_DIR)}`."
    )
    st.stop()

try:
    model, kind = load_model(str(model_path), model_path.stat().st_mtime)
except Exception as exc:  # noqa: BLE001
    st.error(str(exc))
    st.stop()

st.sidebar.success(f"Model dimuat ({kind}): {uploaded_model.name if uploaded_model else model_path.name}")

tab_upload, tab_camera = st.tabs(["📁 Unggah gambar", "📷 Kamera"])
with tab_upload:
    file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png", "webp"])
with tab_camera:
    photo = st.camera_input("Ambil foto")

source = file or photo
if source is None:
    st.stop()

try:
    image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
except Exception:  # noqa: BLE001
    st.error("Gambar tidak dapat dibaca. Coba file lain.")
    st.stop()

col_img, col_res = st.columns(2, gap="large")
with col_img:
    st.subheader("Gambar")
    image_slot = st.empty()
    image_slot.image(image, use_container_width=True)

with col_res:
    st.subheader("Hasil")
    try:
        with st.spinner("Menganalisis..."):
            if kind == "yolo":
                probs, model_names, annotated, detections = predict_yolo(model, image, int(img_size))
                labels = user_labels or model_names
                if probs is not None:
                    show_ranking(probs, labels, top_k)
                else:
                    image_slot.image(annotated, use_container_width=True)
                    if detections:
                        st.dataframe(pd.DataFrame(detections), hide_index=True, use_container_width=True)
                    else:
                        st.info("Tidak ada objek terdeteksi.")
            else:
                probs = predict_torch(model, image, int(img_size), mean, std)
                labels = user_labels
                if labels and len(labels) != len(probs):
                    st.warning(f"Jumlah nama kelas ({len(labels)}) tidak sama dengan jumlah keluaran model ({len(probs)}).")
                show_ranking(probs, labels, top_k)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediksi gagal: {exc}")
        st.caption("Periksa ukuran input dan jenis normalisasi di sidebar.")
