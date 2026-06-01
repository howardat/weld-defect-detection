import ollama
import json
import base64
from pathlib import Path
from weld_pipeline import timing

class WeldAuditor:
    def __init__(self, model_id="gemma4"):
        print(f"--- Using Ollama with {model_id} ---")
        self.model_id = model_id

        self.system_prompt_visual = (
            "You are a Senior ISO 5817 Welding Inspector. Perform a PRIMARY VISUAL INSPECTION. "
            
            "ASSESSMENT STRUCTURE:\n"
            "1. Cracks: [Commentaries about cracks and their location] | Quality level: [B or F]\n"
            
            "2. Porosity: Describe the distribution density (e.g., localized cluster, scattered, or linear) "
            "and estimate the physical diameter of the largest pore. | Quality level: [B, C, D, or F]\n"
            
            "3. Continuity: [Commentaries about discontinuities and their location] | Quality level: [B or F]\n\n"
            
            "FINAL CONCLUSION: Overall ISO Quality Level: [Lowest grade]\n"
            "RULES: Provide qualitative justification based on visual evidence. No recommendations."
        )

        # --- SYSTEM PROMPT B: GROUNDED ---
        self.system_prompt_grounded = (
            """You are a Senior ISO 5817 Welding Inspector.
Your task is to VALIDATE the weld condition using the JSON detections as the only source of defect existence.

ABSOLUTE RULE — DEFECT EXISTENCE

The JSON defines which defects exist.

You are NOT allowed to:

discover new defects

visually infer defects

contradict JSON

If a defect is absent in JSON → it is considered not present and must be justified as acceptable (Level B).

The image is used only to justify the condition, not to determine it.

ANALYSIS BEHAVIOR

For each category (Cracks, Porosity, Continuity):

If present in JSON:
→ Describe visible morphology that supports the defect.

If absent in JSON:
→ Describe visual uniformity and explicitly justify why it satisfies ISO 5817 Level B acceptance.

Never speculate beyond the JSON status.

VISUAL JUSTIFICATION RULES

Cracks → opening shape and orientation
Porosity → surface cleanliness and absence of cavities
Continuity → ripple regularity and uninterrupted bead flow

Descriptions must reference visible weld features, not assumptions.

Maximum 2 sentences per category.

FAILURE RULE

If JSON contains:

any Crack OR

any Discontinuity

→ That category = F

Otherwise evaluate according to ISO acceptance (B/C/D only for Porosity if present).

OUTPUT FORMAT (STRICT)

Cracks: <technical justification>
Quality level: <B or F>

Porosity: <technical justification>
Quality level: <B, C, D, or F>

Continuity: <technical justification>
Quality level: <B or F>

FINAL CONCLUSION:
Overall ISO Quality Level: <lowest grade above>

PROHIBITIONS

Do not:

mention JSON

mention detection process

add defects

give recommendations

omit any section"""
        )    

    def run_single_audit(self, image_path, json_path):
        """Runs both inferences and returns them as strings for main.py."""
        print(f"--- Running Visual-Only Report ---")
        report_visual = self.generate_inference(image_path, json_path=None)
        
        print(f"--- Running Grounded ISO Report ---")
        report_grounded = self.generate_inference(image_path, json_path=json_path)
        
        return report_visual, report_grounded

    def _load_json_context(self, json_path: Path):
        """Formats your specific cleaned JSON (bbox, size, grade) for Gemma."""
        if not json_path.exists():
            return "No automated scan data available for this image."
        
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            if not data: return "No defects detected by pre-processor."
            
            context = "### SCAN_RESULTS\n"
            for entry in data:
                if "discontinuity" in entry:
                    context += f"- Continuity Status: {'FAIL' if entry['discontinuity'] else 'PASS'}\n"
                else:
                    name = entry.get('name', 'DEFECT').upper()
                    bbox = entry.get('bbox', [])
                    size = entry.get('size', 'N/A')
                    grade = entry.get('grade', 'N/A')
                    context += f"- {name}: BBox {bbox}, Size {size}mm, ISO-Grade {grade}\n"
            return context
        except Exception as e:
            return f"Error loading scan data: {str(e)}"

    def generate_inference(self, image_path, json_path=None):
        # Gemma 4 requires images as base64 or bytes in Ollama
        with open(image_path, 'rb') as f:
            image_data = f.read()

        if json_path:
            system_msg = self.system_prompt_grounded
            context = self._load_json_context(Path(json_path))
            user_text = f"{context}\n\nPerform a grounded assessment."
        else:
            system_msg = self.system_prompt_visual
            user_text = "Perform a visual-only assessment of this weld."

        _img_name = Path(image_path).name
        _call_type = "vlm_grounded" if json_path else "vlm_visual"

        # Gemma 4 supports native system roles in Ollama
        with timing.track(_img_name, "vlm", _call_type):
            response = ollama.chat(
                model=self.model_id,
                keep_alive=-1,  # Keep model pinned in VRAM for the entire batch
                messages=[
                    {'role': 'system', 'content': system_msg},
                    {'role': 'user', 'content': user_text, 'images': [image_data]}
                ],
                options={
                    'temperature': 1.0,  # Recommended for Gemma 4
                    'top_p': 0.95,
                    'top_k': 64
                }
            )

        return response['message']['content'].strip()

def run_comparative_batch(image_dir, json_dir, output_dir):
    """Processes all images and generates two reports for each."""
    auditor = WeldAuditor()
    img_dir = Path(image_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])

    for img_path in image_files:
        print(f"\n--- Auditing: {img_path.name} ---")
        json_path = Path(json_dir) / f"{img_path.stem}.json"
        
        # 1. Visual Only
        print("  Running Report A (Visual)...")
        res_a = auditor.generate_inference(img_path, json_path=None)
        
        # 2. Grounded
        print("  Running Report B (Grounded)...")
        res_b = auditor.generate_inference(img_path, json_path=json_path)

        # Save Markdown Comparison
        report_file = out_dir / f"{img_path.stem}_audit.md"
        with open(report_file, "w") as f:
            f.write(f"# Comparative Audit: {img_path.name}\n\n")
            f.write("## 1. Raw Visual Report\n")
            f.write(f"{res_a}\n\n---\n\n")
            f.write("## 2. Grounded Report (YOLO + Metrology Info)\n")
            f.write(f"{res_b}\n")
        
    print(f"\nBatch complete. Reports in: {output_dir}")

if __name__ == "__main__":
    # Example Usage
    run_comparative_batch(
        image_dir="../../data/test_images",
        json_dir="../../data/json_output",
        output_dir="../../reports/gemini_comparisons"
    )