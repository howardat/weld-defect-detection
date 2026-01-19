import torch
import json
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

class WeldAuditor:
    def __init__(self, model_id="google/gemma-3-12b-it"):
        print(f"--- Loading {model_id} ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        
        # 4-bit Quantization for VRAM efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_quant_type="nf4"
        )
        
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation="sdpa"
        )

        self.system_prompt = (
            "You are a highly experienced welding inspector AI. Your task is to provide a detailed, "
            "descriptive assessment of the weld image based on three key criteria.\n\n"
            "The output must be a concise report, formatted with only the main section headings and the "
            "combined assessment text. DO NOT use sub-headings like 'Description' or 'Conclusion' under each section. "
            "The goal is a highly readable report.\n\n"
            "1. Welding Bead Continuity (Start-to-End Consistency)\n"
            "Provide a combined assessment on the weld bead's consistency, noting interruptions, starts/stops, "
            "or profile variations. If you find discontinuities in input data, consider it a gap and the weld cannot be accepted.\n\n"
            "2. Density of Pores (Porosity Assessment)\n"
            "Provide a combined assessment on the presence and severity of pores. Classify density (None, Low, Moderate, High) "
            "and describe distribution.\n\n"
            "3. Relief of the Weld (Smoothness/Bumps)\n"
            "Provide a combined assessment on the weld surface topography, describing its smoothness or the presence of "
            "excessive bumps, unevenness, or sharp transitions.\n\n"
            "Provide a final overall conclusion on the quality.\n\n"
            "Rules:\n"
            "- Start directly with 'ASSESSMENT:'.\n"
            "- Do not use conversational filler or disclaimers.\n"
            "- Do not add recommendations or extra info not found in the scan data."
        )

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
        """Runs the VLM with or without JSON grounding."""
        raw_image = Image.open(image_path).convert("RGB")
        
        # Determine if we are running Grounded or Visual-Only
        context = ""
        if json_path:
            context = self._load_json_context(Path(json_path))
            user_text = f"{context}\n\nPerform a grounded assessment of this weld."
        else:
            user_text = "Perform a visual assessment of this weld without external data."

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": raw_image},
                {"type": "text", "text": user_text}
            ]}
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False, # Greedy search for stability
                use_cache=True
            )
        
        return self.processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

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