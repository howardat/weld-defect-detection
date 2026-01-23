import torch
import json
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

class WeldAuditor:
    def __init__(self, model_id="google/gemma-3-1b-it"):
        print(f"--- Loading {model_id} ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_quant_type="nf4"
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            use_fast=True 
        )
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0}, 
            trust_remote_code=True,
            dtype=self.dtype, # Using 'dtype' instead of 'torch_dtype'
            attn_implementation="sdpa"
        )

        self.system_prompt = (
            "You are a Senior Welding Inspector. Your task is to provide a technical assessment and "
            "quality level for a weld based on ISO 5817 quality levels (B: Stringent, C: Intermediate, D: Moderate).\n\n"
            
            "CRITICAL SAFETY RULE:\n"
            "If 'SCAN_RESULTS' indicate a 'Continuity Status: FAIL' or any 'CRACK' is detected, "
            "the Overall quality level is 'F (FAIL)' regardless of other findings.\n\n"

            "STRUCTURE YOUR RESPONSE AS FOLLOWS:\n"
            "ASSESSMENT:\n"
            "1. Welding Bead Continuity\n"
            "[Text assessment] | Quality level: [A-F]\n\n"
            
            "2. Density of Pores (Porosity)\n"
            "[Text assessment based on ISO grade/size] | Quality level: [A-F]\n\n"
            
            "3. Relief of the Weld (Topography)\n"
            "[Text assessment of smoothness/bumps] | Quality level: [A-F]\n\n"
            
            "FINAL CONCLUSION:\n"
            "Overall ISO Quality Quality level: [A-F] determined by the lowest quality level score mentioned\n\n"

            "RULES:\n"
            "- If SCAN_RESULTS provide an 'ISO-Grade', use it to determine the score.\n"
            "- Start directly with 'ASSESSMENT:'.\n"
            "- Be concise. No filler. No recommendations."
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