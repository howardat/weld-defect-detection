import torch
import json
from PIL import Image
from pathlib import Path
from transformers import AutoModelForCausalLM

_MODEL = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.float16 if _DEVICE == "cuda" else torch.float32


def load_moondream(model_id="vikhyatk/moondream2"):
    global _MODEL
    if _MODEL is None:
        print(f"--- Loading {model_id} ---")
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=_DTYPE,
            device_map=_DEVICE
        )
    return _MODEL


# -------------------------
# Prompt
# -------------------------

SYSTEM_PROMPT = (
    "You are an ISO 5817 welding inspector.\n\n"
    "Look at the weld image and answer the following:\n\n"
    "1. Are there visible cracks? Answer PASS or FAIL and explain briefly.\n"
    "2. Are there visible pores? How many?\n"
    "3. Is the weld bead continuous without breaks? Answer PASS or FAIL.\n\n"
    "Respond in this exact structure:\n\n"
    "Cracks: <PASS or FAIL> – <short explanation>\n"
    "Porosity: <pore COUNT or NONE>\n"
    "Continuity: <PASS or FAIL> – <short explanation>"
)


# -------------------------
# JSON formatter
# -------------------------

def load_json_context(json_path: Path) -> str:
    if not json_path.exists():
        return "No automated scan data available."

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        if not data:
            return "No defects detected by pre-processor."

        lines = ["SCAN RESULTS:"]
        for entry in data:
            if "discontinuity" in entry:
                lines.append(
                    f"- Continuity: {'FAIL' if entry['discontinuity'] else 'PASS'}"
                )
            else:
                name = entry.get("name", "DEFECT").upper()
                bbox = entry.get("bbox", [])
                size = entry.get("size", "N/A")
                grade = entry.get("grade", "N/A")
                lines.append(
                    f"- {name}: BBox {bbox}, Size {size}mm, ISO Grade {grade}"
                )

        return "\n".join(lines)

    except Exception as e:
        return f"Error loading scan data: {e}"


# -------------------------
# Core inference
# -------------------------

def run_audit(image_path: Path, json_path: Path | None = None):
    model = load_moondream()

    image = Image.open(image_path).convert("RGB")

    # 🔥 Encode ONCE
    with torch.inference_mode():
        image_embeds = model.encode_image(image)

    # ---- Visual ----
    visual_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Perform a visual assessment of this weld without external data."
    )

    with torch.inference_mode():
        visual_report = model.answer_question(
            image_embeds=image_embeds,
            question=visual_prompt,
            max_tokens=1000
        ).strip()

    # ---- Grounded ----
    if json_path:
        context = load_json_context(json_path)
        grounded_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            # f"{context}\n\n"
            "Perform a grounded assessment of this weld."
        )

        with torch.inference_mode():
            grounded_report = model.answer_question(
                image_embeds=image_embeds,
                question=grounded_prompt,
                max_tokens=1000
            ).strip()
    else:
        grounded_report = None

    print(grounded_report)
    return visual_report, grounded_report




if __name__ == "__main__":
    pass