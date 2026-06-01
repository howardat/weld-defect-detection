from ultralytics import YOLO

MODEL_PATH = '/Users/oljk/Projects/weld-pipeline/models/best.pt'
model = YOLO(MODEL_PATH)
results = model.predict('../../data/img/572100515_3005304273003849_8602745394797406488_n.jpg', conf=0.05, classes=3, verbose=False, save=False)