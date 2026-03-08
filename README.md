# Object Detection Project

## Cấu trúc thư mục
- `configs/`: File cấu hình cho mô hình, dữ liệu và huấn luyện.
- `datasets/`: Code xử lý dữ liệu và biến đổi.
- `models/`: Định nghĩa mô hình (backbone, head, loss).
- `training/`: Code huấn luyện.
- `evaluation/`: Code đánh giá mô hình.
- `utils/`: Công cụ hỗ trợ (logger, visualization, seed).
- `outputs/`: Lưu trọng số mô hình.
- `tests/`: Kiểm tra dataset, mô hình và huấn luyện.

## Hướng dẫn sử dụng
1. **Chuẩn bị dữ liệu**:
   - Đặt ảnh vào `data/train` và `data/val`.
   - Chuyển file `.csv` thành nhãn YOLO trong `data/labels/train` và `data/labels/val`.
2. **Cài đặt**:
   ```bash
   pip install -r requirements.txt