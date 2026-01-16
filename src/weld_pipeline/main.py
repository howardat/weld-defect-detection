import cv2
from detector import WeldDetector
from visualizer import plot_multi_stage_results

def main():
    detector = WeldDetector("../../models/best.pt") # Ensure your model path is correct
    image = cv2.imread("../../data/zoom.jpg")
    if image is None: return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.run_inference(image_rgb)
    
    if results:
        plot_multi_stage_results(results)

if __name__ == "__main__":
    main()