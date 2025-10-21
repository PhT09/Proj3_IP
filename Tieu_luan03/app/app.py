import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from feature_funcs import (
    threshold_global, threshold_adaptive_mean, threshold_adaptive_gaussian,
    threshold_otsu, region_growing, watershed_segmentation,
    connected_components_detection, contour_detection, boundary_representation
)
from supporting_functions import (
    load_and_convert_image, resize_for_display, 
    overlay_edges_on_original, convert_to_pil
)

# Cấu hình trang
st.set_page_config(page_title="Xử lý ảnh - Segmentation", layout="wide")

# CSS để giảm khoảng cách và tối ưu layout
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    h1 {margin-bottom: 0.5rem; font-size: 1.8rem;}
    h2 {margin-top: 0.5rem; margin-bottom: 0.5rem; font-size: 1.3rem;}
    h3 {margin-top: 0.3rem; margin-bottom: 0.3rem; font-size: 1.1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
</style>
""", unsafe_allow_html=True)

st.title("🖼️ Xử lý ảnh - Phân đoạn ảnh (Segmentation)")

# Khởi tạo session state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'chain_codes' not in st.session_state:
    st.session_state.chain_codes = None

# Layout chính
col_control, col_display = st.columns([1, 2])

with col_control:
    st.header("Điều khiển")
    
    # Upload ảnh
    uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        st.session_state.original_image = load_and_convert_image(uploaded_file)
        st.success("✅ Đã tải ảnh thành công!")
    
    # Nút xóa ảnh
    if st.button("🗑️ Xóa ảnh hiện tại", use_container_width=True):
        st.session_state.original_image = None
        st.session_state.processed_image = None
        st.session_state.chain_codes = None
        st.rerun()
    
    st.divider()
    
    # Chọn chức năng
    if st.session_state.original_image is not None:
        st.subheader("Chọn chức năng")
        
        function = st.selectbox(
            "Phương pháp",
            ["Làm sắc nét", "Phân đoạn ngưỡng toàn cục", "Phân đoạn ngưỡng thích nghi (Trung bình)",
             "Phân đoạn ngưỡng thích nghi (Gaussian)", "Phân đoạn ngưỡng Otsu",
             "Phân đoạn dựa trên vùng (Region Growing)", "Phân đoạn Watershed",
             "Xác định đối tượng (Connected Components)", "Xác định đối tượng (Contour)",
             "Biểu diễn biên (Chain Code)"],
            label_visibility="collapsed"
        )
        
        # Reset chain_codes khi đổi chức năng
        if function != "Biểu diễn biên (Chain Code)":
            st.session_state.chain_codes = None
        
        # Reset seed_point khi không phải Region Growing
        if function != "Phân đoạn dựa trên vùng (Region Growing)":
            if 'seed_point' in st.session_state:
                st.session_state.seed_point = None
        
        # Tham số và xử lý tự động cho từng chức năng
        if function == "Phân đoạn ngưỡng toàn cục":
            threshold = st.slider("Ngưỡng", 0, 255, 127)
            
            # Xử lý tự động
            st.session_state.processed_image = threshold_global(st.session_state.original_image, threshold)
        
        elif function == "Phân đoạn ngưỡng thích nghi (Trung bình)":
            block_size = st.slider("Block size", 3, 51, 11, 2)
            c_value = st.slider("C", -10, 10, 2)
            
            # Xử lý tự động
            st.session_state.processed_image = threshold_adaptive_mean(st.session_state.original_image, block_size, c_value)
        
        elif function == "Phân đoạn ngưỡng thích nghi (Gaussian)":
            block_size = st.slider("Block size", 3, 51, 11, 2)
            c_value = st.slider("C", -10, 10, 2)
            
            # Xử lý tự động
            st.session_state.processed_image = threshold_adaptive_gaussian(st.session_state.original_image, block_size, c_value)
        
        elif function == "Phân đoạn ngưỡng Otsu":
            st.info("Không cần tham số - Tự động xác định ngưỡng tối ưu")
            
            # Xử lý tự động
            st.session_state.processed_image = threshold_otsu(st.session_state.original_image)
        
        elif function == "Phân đoạn dựa trên vùng (Region Growing)":
            st.write("Chọn seed point:")
            seed_x = st.slider("Seed X", 0, st.session_state.original_image.shape[1]-1, st.session_state.original_image.shape[1]//2)
            seed_y = st.slider("Seed Y", 0, st.session_state.original_image.shape[0]-1, st.session_state.original_image.shape[0]//2)
            threshold = st.slider("Ngưỡng", 1, 50, 10)
            
            # Lưu seed point vào session state
            st.session_state.seed_point = (seed_x, seed_y)
            
            # Xử lý tự động
            st.session_state.processed_image = region_growing(st.session_state.original_image, seed_x, seed_y, threshold)
        
        elif function == "Phân đoạn Watershed":
            morph_kernel = st.slider("Morphology kernel size", 3, 15, 5, 2)
            
            # Xử lý tự động
            st.session_state.processed_image = watershed_segmentation(st.session_state.original_image, morph_kernel)
        
        elif function == "Xác định đối tượng (Connected Components)":
            min_area = st.slider("Diện tích tối thiểu (pixels)", 10, 1000, 100, 10)
            
            # Xử lý tự động
            st.session_state.processed_image = connected_components_detection(st.session_state.original_image, min_area)
        
        elif function == "Xác định đối tượng (Contour)":
            min_area = st.slider("Diện tích tối thiểu (pixels)", 10, 1000, 100, 10)
            
            # Xử lý tự động
            st.session_state.processed_image = contour_detection(st.session_state.original_image, min_area)
        
        elif function == "Biểu diễn biên (Chain Code)":
            min_area = st.slider("Diện tích contour tối thiểu (pixels)", 10, 1000, 100, 10)
            
            # Xử lý tự động
            result, chain_info = boundary_representation(st.session_state.original_image, min_area)
            st.session_state.processed_image = result
            st.session_state.chain_codes = chain_info
        
        # Nút tải xuống
        if st.session_state.processed_image is not None:
            st.divider()
            pil_img = convert_to_pil(st.session_state.processed_image)
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Tải ảnh kết quả",
                data=buf.getvalue(),
                file_name="result.png",
                mime="image/png",
                use_container_width=True
            )

# Hiển thị ảnh
with col_display:
    st.header("Hiển thị ảnh")
    
    if st.session_state.original_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ảnh gốc")
            display_img = resize_for_display(st.session_state.original_image, max_width=400)
            
            # Nếu đang dùng Region Growing, vẽ seed point
            if 'seed_point' in st.session_state and st.session_state.seed_point is not None:
                # Tạo bản sao để vẽ
                img_with_seed = display_img.copy()
                if len(img_with_seed.shape) == 2:
                    img_with_seed = cv2.cvtColor(img_with_seed, cv2.COLOR_GRAY2BGR)
                
                # Tính tỷ lệ resize để đặt seed point đúng vị trí
                original_h, original_w = st.session_state.original_image.shape[:2]
                display_h, display_w = display_img.shape[:2]
                scale_x = display_w / original_w
                scale_y = display_h / original_h
                
                # Vẽ seed point (chấm tròn đỏ và chữ thập)
                seed_x, seed_y = st.session_state.seed_point
                display_seed_x = int(seed_x * scale_x)
                display_seed_y = int(seed_y * scale_y)

                # Vẽ điểm nhỏ ở giữa
                cv2.circle(img_with_seed, (display_seed_x, display_seed_y), 2, (0, 0, 255), -1)
                
                st.image(img_with_seed, use_container_width=True, clamp=True)
            else:
                st.image(display_img, use_container_width=True, clamp=True)
        
        with col2:
            st.subheader("Ảnh kết quả")
            if st.session_state.processed_image is not None:
                display_result = resize_for_display(st.session_state.processed_image, max_width=400)
                st.image(display_result, use_container_width=True, clamp=True)
            else:
                st.info("Chọn chức năng để xem kết quả")
        
        # Hiển thị chain code nếu có
        if st.session_state.chain_codes is not None and len(st.session_state.chain_codes) > 0:
            st.divider()
            st.subheader("Kết quả Chain Code")
            for info in st.session_state.chain_codes:
                with st.expander(f"Contour {info['contour_id']} - Diện tích: {info['area']:.0f}px"):
                    st.write(f"**Độ dài chain code:** {info['chain_length']}")
                    st.write(f"**Chain code (50 số đầu):** {info['chain_code'][:50]}")
                    if info['chain_length'] > 50:
                        st.caption(f"... và {info['chain_length'] - 50} số nữa")
    else:
        st.info("📤 Vui lòng tải ảnh lên để bắt đầu")