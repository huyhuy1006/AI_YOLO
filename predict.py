import os
import cv2
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path
import numpy as np

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config {config_path}: {e}")

def draw_bounding_boxes(image: np.ndarray, results: list, class_names: list) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    for result in results:
        for box in result.boxes:
            # Extract box coordinates, confidence, and class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            cls = int(box.cls.item())
            label = f"{class_names[cls]} {conf:.2f}"
            
            # Draw rectangle and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
    return image

def save_image(image: np.ndarray, output_path: str) -> None:
    """Save the image to the specified path."""
    try:
        cv2.imwrite(output_path, image)
    except Exception as e:
        print(f"Failed to save image {output_path}: {e}")

def predict_and_visualize(
    model_path: str,
    test_dir: str,
    output_dir: str,
    img_size: int,
    class_names: list,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> None:
    """Perform inference on test images and save results with bounding boxes."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = YOLO(model_path).to(device)
    
    # Get list of test images
    test_images = [f for f in Path(test_dir).glob('*.jpg')]
    if not test_images:
        raise ValueError(f"No images found in {test_dir}")
    
    # Perform inference
    for img_path in test_images:
        # Load and predict
        results = model.predict(
            source=str(img_path),
            imgsz=img_size,
            device=device,
            conf=0.5  # Confidence threshold
        )
        
        # Load image with OpenCV
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load image {img_path}")
            continue
        
        # Draw bounding boxes
        image = draw_bounding_boxes(image, results, class_names)
        
        # Save output image
        output_path = os.path.join(output_dir, f"pred_{img_path.name}")
        save_image(image, output_path)
        print(f"Saved prediction for {img_path} to {output_path}")

def main():
    # Load configurations
    data_config = load_config('configs/data.yaml')
    train_config = load_config('configs/train.yaml')
    
    # Define paths and parameters
    model_path = 'outputs/yolov8/yolov8n_car/weights/best.pt'
    test_dir = str(Path(data_config['test']).joinpath('images'))
    output_dir = 'outputs/yolov8/yolov8n_car/predictions'
    img_size = train_config['img_size']
    class_names = data_config['names']
    
    # Run prediction and visualization
    predict_and_visualize(model_path, test_dir, output_dir, img_size, class_names)

if __name__ == '__main__':
    main()