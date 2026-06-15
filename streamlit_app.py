import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st
import json
import numpy as np
from PIL import Image
from ultralytics import YOLO
import subprocess
import sys

from weld_pipeline.main import (DISC_THRESHOLD, OTSU_MULTIPLIER, PX_TO_MM, THROAT_THICKNESS,
                                MARKER_SIZE_MM, MIN_PORE_SIZE_MM, PORE_OTSU_MULT,
                                PORE_CLUSTER_DIST_MM, USE_VLM)


def pick_folder(initial_dir: str) -> str:
    script = (
        "import tkinter as tk\nfrom tkinter import filedialog\n"
        "root = tk.Tk()\nroot.withdraw()\nroot.wm_attributes('-topmost', 1)\n"
        f"print(filedialog.askdirectory(initialdir={initial_dir!r}) or '')\n"
        "root.destroy()\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    return result.stdout.strip()

st.set_page_config(page_title="Weld Inspection Pipeline", layout="wide")
st.title("Weld Inspection Pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "wda11s-seg.pt"
DEFAULT_IMAGE_DIR = str(PROJECT_ROOT / "data" / "discontinuity")


@st.cache_resource
def load_seg_model():
    return YOLO(MODEL_PATH)


def _ollama_available() -> bool:
    try:
        import importlib
        importlib.import_module("ollama")
        return True
    except ImportError:
        return False

_VLM_AVAILABLE = _ollama_available()

@st.cache_resource
def load_auditor():
    from weld_pipeline.vlm import WeldAuditor
    return WeldAuditor()



# --- Sidebar ---
if "image_folder" not in st.session_state:
    st.session_state.image_folder = DEFAULT_IMAGE_DIR

with st.sidebar:
    st.header("Configuration")
    path_col, browse_col = st.columns([4, 1])
    with path_col:
        image_folder = st.text_input("Image folder", value=st.session_state.image_folder,
                                     key="image_folder_input")
    with browse_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📂"):
            chosen = pick_folder(st.session_state.image_folder)
            if chosen:
                st.session_state.image_folder = chosen
                st.rerun()
    image_folder = st.session_state.image_folder
    throat_thickness = st.number_input("Throat thickness (mm)", value=float(THROAT_THICKNESS), min_value=0.1, step=0.5)
    use_intensity_filter = st.toggle("Porosity intensity filter", value=True)
    use_vlm = st.toggle("Enable VLM analysis", value=USE_VLM and _VLM_AVAILABLE, disabled=not _VLM_AVAILABLE)
    if not _VLM_AVAILABLE:
        st.caption("ollama not installed — VLM unavailable")
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

# --- Analysis ---
if run_btn:
    from weld_pipeline.discontinuity_check import discontinuity_check
    from weld_pipeline.porosity_check import porosity_check
    from weld_pipeline.cracks_check import cracks_check
    from weld_pipeline.visualization import draw_technical_overlay

    image_dir = Path(image_folder)
    if not image_dir.is_dir():
        st.error(f"Directory not found: {image_folder}")
        st.stop()

    all_images = sorted([
        p for ext in ("*.jpeg", "*.jpg", "*.png")
        for p in image_dir.glob(ext)
    ])

    if not all_images:
        st.warning("No images found in the specified directory.")
        st.stop()

    seg_model = load_seg_model()
    auditor = load_auditor() if use_vlm else None


    json_dir = PROJECT_ROOT / "data" / "json_output"
    json_dir.mkdir(parents=True, exist_ok=True)

    st.info(f"Found {len(all_images)} image(s). Processing...")
    progress = st.progress(0)

    for idx, image_path in enumerate(all_images):
        with st.spinner(f"Processing {image_path.name}…"):
            disc_bool, refined_masks, line_params = discontinuity_check(
                image_path=str(image_path),
                model_path=str(MODEL_PATH),
                threshold=DISC_THRESHOLD,
                otsu_multiplier=OTSU_MULTIPLIER,
                visualize=False,
                seg_model=seg_model,
            )

            clean_json_list, raw_pore_data, _ = porosity_check(
                image_path=str(image_path),
                model_path=str(MODEL_PATH),
                px_to_mm=PX_TO_MM,
                throat_thickness=throat_thickness,
                marker_size_mm=MARKER_SIZE_MM,
                otsu_multiplier=PORE_OTSU_MULT,
                min_pore_size_mm=MIN_PORE_SIZE_MM,
                pore_cluster_distance_mm=PORE_CLUSTER_DIST_MM,
                use_intensity_filter=use_intensity_filter,
                visualize=False,
                seg_model=seg_model,
            )

            _, crack_masks_list, all_crack_detections, weld_mask = cracks_check(
                image_path=str(image_path),
                model_path=str(MODEL_PATH),
                seg_model=seg_model,
            )

            overlay_kwargs = dict(
                line_params=line_params,
                discontinuity_masks=refined_masks,
                weld_mask=weld_mask,
                porosity_data=raw_pore_data,
                crack_masks=crack_masks_list,
                disc_bool=disc_bool,
            )

            img_original = np.array(Image.open(image_path).convert("RGB"))
            img_labeled = draw_technical_overlay(str(image_path), show_labels=True, **overlay_kwargs)
            img_clean = draw_technical_overlay(str(image_path), show_labels=False, **overlay_kwargs)

            if use_vlm and auditor is not None:
                image_json_path = json_dir / f"{image_path.stem}.json"
                with open(image_json_path, "w") as f:
                    json.dump([{"discontinuity": disc_bool}] + all_crack_detections + clean_json_list, f, indent=4)
                report_v, report_g = auditor.run_single_audit(image_path, image_json_path)
            else:
                report_v, report_g = None, None

        st.markdown(f"### {image_path.name}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_original, caption="Original", use_container_width=True)
        with col2:
            st.image(img_labeled, caption="Detections + Labels", use_container_width=True)
        with col3:
            st.image(img_clean, caption="Detections (no labels)", use_container_width=True)

        with st.expander("Inference results", expanded=True):
            det_col, vlm_col = st.columns(2)

            with det_col:
                st.markdown("**Detection summary**")
                disc_label = ":red[FAIL]" if disc_bool else ":green[PASS]"
                st.markdown(f"- Discontinuity: {disc_label}")
                st.markdown(f"- Cracks: `{len(all_crack_detections)}`")
                pore_str = ", ".join(f"{p.get('grade','?')} {p.get('size','?')}mm" for p in clean_json_list)
                st.markdown(f"- Pores `{len(clean_json_list)}`" + (f": `{pore_str}`" if pore_str else ""))
                if line_params:
                    angles_deg = [
                        f"{np.degrees(lp['angle']):.1f}°"
                        for lp in line_params if lp.get("angle") is not None
                    ]
                    if angles_deg:
                        st.markdown(f"- Segment angles: `{', '.join(angles_deg)}`")

            with vlm_col:
                if use_vlm and report_g is not None:
                    st.markdown("**Grounded report**")
                    st.text(report_g)
                else:
                    st.markdown("*VLM disabled*")

        st.divider()
        progress.progress((idx + 1) / len(all_images))

    st.success(f"Done — processed {len(all_images)} image(s).")
