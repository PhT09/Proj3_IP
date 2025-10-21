import cv2
import numpy as np

def threshold_global(image, threshold_value):
    """Phân đoạn ngưỡng toàn cục"""
    _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary

def threshold_adaptive_mean(image, block_size, c_value):
    """Phân đoạn ngưỡng thích nghi - Trung bình"""
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, block_size, c_value)
    return binary

def threshold_adaptive_gaussian(image, block_size, c_value):
    """Phân đoạn ngưỡng thích nghi - Gaussian"""
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, block_size, c_value)
    return binary

def threshold_otsu(image):
    """Phân đoạn ngưỡng Otsu"""
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def region_growing(image, seed_x, seed_y, threshold):
    """Phân đoạn dựa trên vùng - Region Growing"""
    h, w = image.shape
    segmented = np.zeros_like(image)
    visited = np.zeros((h, w), dtype=bool)
    
    seed_value = int(image[seed_y, seed_x])
    stack = [(seed_x, seed_y)]
    visited[seed_y, seed_x] = True
    
    while stack:
        x, y = stack.pop()
        segmented[y, x] = 255
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                if abs(int(image[ny, nx]) - seed_value) <= threshold:
                    stack.append((nx, ny))
                    visited[ny, nx] = True
    
    return segmented

def watershed_segmentation(image, morph_kernel_size):
    """Phân đoạn bằng Watershed"""
    # Chuyển sang nhị phân bằng Otsu
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Loại nhiễu
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Xác định vùng nền chắc chắn
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Xác định vùng foreground
    kernel_dist = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    
    # Tìm vùng chưa xác định
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Gán nhãn
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Áp dụng watershed
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    
    # Tạo ảnh kết quả
    result = np.zeros_like(image)
    result[markers > 1] = 255
    
    return result

def connected_components_detection(image, min_area):
    """Xác định đối tượng bằng Connected Components"""
    # Chuyển sang nhị phân
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Tìm connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Tạo ảnh màu để hiển thị
    output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    # Vẽ các components có diện tích lớn hơn min_area
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background màu đen
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            output[labels == i] = colors[i]
    
    return output

def contour_detection(image, min_area):
    """Xác định đối tượng bằng Contour"""
    # Chuyển sang nhị phân
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Tìm contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tạo ảnh màu để vẽ contours
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Vẽ các contours có diện tích lớn hơn min_area
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
            
            # Vẽ bounding box
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return output

def freeman_chain_code(contour):
    """Tính toán Freeman chain code từ contour"""
    directions = {
        (1, 0): 0,    # Right
        (1, 1): 1,    # Right-Down
        (0, 1): 2,    # Down
        (-1, 1): 3,   # Left-Down
        (-1, 0): 4,   # Left
        (-1, -1): 5,  # Left-Up
        (0, -1): 6,   # Up
        (1, -1): 7    # Right-Up
    }
    
    chain_code = []
    contour = contour.squeeze()
    
    if len(contour.shape) == 1:
        return []
    
    for i in range(len(contour) - 1):
        dx = contour[i+1][0] - contour[i][0]
        dy = contour[i+1][1] - contour[i][1]
        
        # Chuẩn hóa về -1, 0, 1
        dx = np.sign(dx)
        dy = np.sign(dy)
        
        if (dx, dy) in directions:
            chain_code.append(directions[(dx, dy)])
    
    return chain_code

def boundary_representation(image, min_contour_area):
    """Biểu diễn biên bằng Chain Code"""
    # Chuyển sang nhị phân
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Tìm contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Tạo ảnh output
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    chain_codes_info = []
    
    for idx, contour in enumerate(contours):
        if cv2.contourArea(contour) >= min_contour_area:
            # Vẽ contour
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
            
            # Tính chain code
            chain_code = freeman_chain_code(contour)
            
            if len(chain_code) > 0:
                chain_codes_info.append({
                    'contour_id': idx,
                    'area': cv2.contourArea(contour),
                    'chain_code': chain_code,
                    'chain_length': len(chain_code)
                })
    
    return output, chain_codes_info