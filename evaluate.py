# evaluate.py
from ultralytics import YOLO
import yaml

def main():
    with open('configs/data.yaml', 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    with open('configs/train.yaml', 'r', encoding='utf-8') as f:
        train_config = yaml.safe_load(f)

    # Load YOLOv8 model
    model = YOLO('outputs/yolov8/yolov8n_car/weights/best.pt')

    # Evaluate on validation set
    results = model.val(
        data='configs/data.yaml',
        imgsz=train_config['img_size'],
        batch=train_config['batch_size'],
        device=0  # Use GPU
    )

    # Print metrics
    print(f"mAP@0.5: {results.box.map50}, mAP@0.5:0.95: {results.box.map}, "
          f"Precision: {results.box.p}, Recall: {results.box.r}, "
          f"F1 Score: {2 * (results.box.p * results.box.r) / (results.box.p + results.box.r + 1e-6)}")

if __name__ == '__main__':
    main()