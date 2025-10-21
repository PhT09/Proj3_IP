import cv2
import numpy as np
from PIL import Image

def load_and_convert_image(uploaded_file):
    """Đọc file ảnh và chuyển sang grayscale"""
    # Đọc file
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Chuyển sang grayscale nếu là ảnh màu
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
    
    return gray_image

def resize_for_display(image, max_width=500):
    """Resize ảnh để hiển thị, giữ nguyên tỷ lệ"""
    h, w = image.shape[:2]
    
    if w > max_width:
        ratio = max_width / w
        new_w = max_width
        new_h = int(h * ratio)
        
        if len(image.shape) == 2:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return resized
    
    return image

def overlay_edges_on_original(original, edges):
    """Chồng biên lên ảnh gốc"""
    # Chuyển ảnh gốc sang màu nếu cần
    if len(original.shape) == 2:
        color_original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        color_original = original.copy()
    
    # Chuyển edges sang màu nếu cần
    if len(edges.shape) == 2:
        # Tạo mask từ edges
        mask = edges > 0
        color_original[mask] = [0, 255, 0]  # Biên màu xanh lá
    else:
        # Nếu edges đã là ảnh màu (như từ contour detection)
        mask = np.any(edges != [0, 0, 0], axis=-1)
        color_original[mask] = edges[mask]
    
    return color_original

def convert_to_pil(image):
    """Chuyển numpy array sang PIL Image để tải xuống"""
    if len(image.shape) == 2:
        return Image.fromarray(image)
    else:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))