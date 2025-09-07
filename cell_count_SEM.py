import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import requests
import os
from scipy import ndimage
from skimage import measure, morphology

# Page configuration
st.set_page_config(
    page_title="Cell Counter with SAM",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'sam_model' not in st.session_state:
    st.session_state.sam_model = None

@st.cache_resource
def load_sam_model():
    """Load SAM model with caching"""
    try:
        # Download checkpoint if not exists
        checkpoint_path = "sam_vit_b_01ec64.pth"
        if not os.path.exists(checkpoint_path):
            with st.spinner("Downloading SAM model (this is a one-time download, ~375MB)..."):
                url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                progress_bar = st.progress(0)
                downloaded = 0
                
                with open(checkpoint_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress_bar.progress(downloaded / total_size)
                
                progress_bar.empty()
        
        # Load model
        with st.spinner("Loading SAM model..."):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
            sam.to(device=device)
            
            # Create mask generator with parameters optimized for cells
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,  # Higher for better detection of small objects
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Minimum area for detected regions
            )
            
        return mask_generator
    except Exception as e:
        st.error(f"Error loading SAM model: {str(e)}")
        return None

def process_with_sam(image, mask_generator, min_area, max_area, circularity_threshold):
    """Process image with SAM and filter results"""
    # Generate masks
    masks = mask_generator.generate(image)
    
    # Filter masks by area and circularity
    filtered_masks = []
    for mask_data in masks:
        mask = mask_data['segmentation']
        area = mask_data['area']
        
        # Filter by area
        if min_area <= area <= max_area:
            # Calculate circularity
            contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = contours[0]
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity >= circularity_threshold:
                        filtered_masks.append(mask_data)
    
    return filtered_masks

def process_with_traditional(image, blur_size, threshold_value, min_area, max_area):
    """Fallback to traditional CV methods"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply processing
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # Try adaptive threshold for better results
    binary = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            filtered_contours.append(contour)
    
    return filtered_contours, binary

# Title and description
st.title("🔬 Advanced Cell Counter")
st.markdown("""
Automated cell counting using Meta's Segment Anything Model (SAM) for robust segmentation.
SAM provides better performance across variable image conditions.
""")

# Create main layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📁 Upload Images")
    uploaded_files = st.file_uploader(
        "Choose image files", 
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        accept_multiple_files=True,
        help="You can select multiple images at once"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} image(s) uploaded")
        
        with st.expander("View uploaded files"):
            for file in uploaded_files:
                st.text(f"📄 {file.name}")
    
    st.header("⚙️ Segmentation Method")
    
    use_sam = st.radio(
        "Select method:",
        ["SAM (AI-based) - Recommended", "Traditional (Threshold-based)"],
        help="SAM works better for variable images"
    ) == "SAM (AI-based) - Recommended"
    
    st.header("🎯 Detection Parameters")
    
    if use_sam:
        st.markdown("*SAM-specific parameters*")
        
        min_area = st.slider(
            "Min Cell Size (px)", 
            min_value=50, 
            max_value=1000, 
            value=100,
            help="Minimum cell area"
        )
        
        max_area = st.slider(
            "Max Cell Size (px)", 
            min_value=500, 
            max_value=10000, 
            value=5000,
            help="Maximum cell area"
        )
        
        circularity_threshold = st.slider(
            "Circularity Filter",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="0 = any shape, 1 = perfect circle"
        )
        
    else:
        st.markdown("*Traditional CV parameters*")
        
        blur_size = st.slider(
            "Blur Amount", 
            min_value=1, 
            max_value=21, 
            value=5, 
            step=2
        )
        
        threshold_value = st.slider(
            "Threshold Level", 
            min_value=0, 
            max_value=255, 
            value=127
        )
        
        min_area = st.slider(
            "Min Cell Size (px)", 
            min_value=10, 
            max_value=500, 
            value=50
        )
        
        max_area = st.slider(
            "Max Cell Size (px)", 
            min_value=100, 
            max_value=5000, 
            value=2000
        )
    
    # Display options
    st.header("🎨 Display Options")
    show_numbers = st.checkbox("Show cell numbers", value=True)
    show_contours = st.checkbox("Show cell outlines", value=True)
    show_centers = st.checkbox("Show cell centers", value=True)

with col2:
    st.header("🔬 Process & Results")
    
    if uploaded_files:
        col_process, col_download = st.columns([1, 1])
        
        with col_process:
            if st.button("🚀 Process All Images", type="primary", use_container_width=True):
                # Clear previous results
                st.session_state.results = []
                st.session_state.processed_images = []
                
                # Load SAM model if needed
                mask_generator = None
                if use_sam:
                    mask_generator = load_sam_model()
                    if mask_generator is None:
                        st.error("Failed to load SAM model. Falling back to traditional method.")
                        use_sam = False
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    try:
                        # Read image
                        image = Image.open(uploaded_file)
                        img_array = np.array(image)
                        original_img = img_array.copy()
                        
                        # Ensure RGB
                        if len(img_array.shape) == 2:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                            original_img = img_array.copy()
                        
                        result_img = img_array.copy()
                        
                        if use_sam and mask_generator:
                            # Process with SAM
                            masks = process_with_sam(img_array, mask_generator, 
                                                    min_area, max_area, 
                                                    circularity_threshold)
                            
                            # Draw results
                            for i, mask_data in enumerate(masks):
                                mask = mask_data['segmentation']
                                
                                # Find contours for drawing
                                contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                                              cv2.RETR_EXTERNAL, 
                                                              cv2.CHAIN_APPROX_SIMPLE)
                                
                                if contours and show_contours:
                                    # Random color for each cell
                                    color = tuple(np.random.randint(100, 255, 3).tolist())
                                    cv2.drawContours(result_img, contours, -1, color, 2)
                                
                                # Get center
                                if show_centers or show_numbers:
                                    bbox = mask_data['bbox']
                                    cx = int(bbox[0] + bbox[2] / 2)
                                    cy = int(bbox[1] + bbox[3] / 2)
                                    
                                    if show_centers:
                                        cv2.circle(result_img, (cx, cy), 4, (255, 0, 0), -1)
                                    
                                    if show_numbers:
                                        cv2.putText(result_img, str(i+1), 
                                                  (cx+7, cy-7), cv2.FONT_HERSHEY_SIMPLEX, 
                                                  0.6, (255, 255, 0), 2)
                            
                            cell_count = len(masks)
                            binary = None  # SAM doesn't produce binary image
                            
                        else:
                            # Process with traditional method
                            contours, binary = process_with_traditional(
                                img_array, blur_size, threshold_value, 
                                min_area, max_area
                            )
                            
                            # Draw results
                            for i, contour in enumerate(contours):
                                if show_contours:
                                    cv2.drawContours(result_img, [contour], -1, (0, 255, 0), 2)
                                
                                # Get center
                                M = cv2.moments(contour)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    
                                    if show_centers:
                                        cv2.circle(result_img, (cx, cy), 4, (255, 0, 0), -1)
                                    
                                    if show_numbers:
                                        cv2.putText(result_img, str(i+1), 
                                                  (cx+7, cy-7), cv2.FONT_HERSHEY_SIMPLEX, 
                                                  0.6, (255, 255, 0), 2)
                            
                            cell_count = len(contours)
                        
                        # Store result
                        st.session_state.results.append({
                            'Filename': uploaded_file.name,
                            'Cell_Count': cell_count
                        })
                        
                        # Store images for display
                        st.session_state.processed_images.append({
                            'filename': uploaded_file.name,
                            'original': original_img,
                            'segmented': result_img,
                            'binary': binary,
                            'count': cell_count
                        })
                        
                    except Exception as e:
                        st.session_state.results.append({
                            'Filename': uploaded_file.name,
                            'Cell_Count': f"Error: {str(e)}"
                        })
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ Processing complete!")
        
        with col_download:
            if st.session_state.results:
                df = pd.DataFrame(st.session_state.results)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV Results",
                    data=csv,
                    file_name="cell_counts.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )
    
    # Display results
    if st.session_state.results:
        st.markdown("---")
        df = pd.DataFrame(st.session_state.results)
        valid_counts = df[df['Cell_Count'].apply(lambda x: isinstance(x, int))]
        
        if not valid_counts.empty:
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Total Images", len(df))
            with col_stat2:
                st.metric("Total Cells", valid_counts['Cell_Count'].sum())
            with col_stat3:
                st.metric("Average/Image", f"{valid_counts['Cell_Count'].mean():.1f}")
            with col_stat4:
                st.metric("Std Dev", f"{valid_counts['Cell_Count'].std():.1f}")
        
        st.subheader("📊 Results Summary")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Visual results
        st.markdown("---")
        st.subheader("🖼️ Visual Verification")
        
        if st.session_state.processed_images:
            image_names = [img['filename'] for img in st.session_state.processed_images]
            selected_image = st.selectbox(
                "Select image to review:",
                options=range(len(image_names)),
                format_func=lambda x: f"{image_names[x]} ({st.session_state.processed_images[x]['count']} cells)"
            )
            
            img_data = st.session_state.processed_images[selected_image]
            
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                st.markdown("**Original Image**")
                st.image(img_data['original'], use_column_width=True)
            
            with img_col2:
                st.markdown(f"**Segmentation Result** ({img_data['count']} cells)")
                st.image(img_data['segmented'], use_column_width=True)
            
            if img_data['binary'] is not None:
                with st.expander("View threshold image"):
                    st.image(img_data['binary'], caption="Binary threshold", use_column_width=True)
        
        if st.button("🔄 Clear All & Start New Batch", use_container_width=True):
            st.session_state.results = []
            st.session_state.processed_images = []
            st.rerun()
    else:
        st.info("👈 Upload images and click 'Process All Images' to get started")

# Instructions
with st.expander("📖 How to Use"):
    st.markdown("""
    ### Methods:
    - **SAM (Recommended)**: AI-based segmentation that adapts to various image types
    - **Traditional**: Threshold-based method for simpler, uniform images
    
    ### SAM Advantages:
    - Better handles variable lighting and contrast
    - More accurate cell boundary detection
    - Works well with touching or overlapping cells
    - Adapts to different cell types and staining methods
    
    ### Parameters:
    - **Min/Max Size**: Filter detected objects by area
    - **Circularity** (SAM only): Filter by shape (higher = more circular)
    
    Note: First run with SAM will download the model (~375MB, one-time only)
    """)

st.markdown("---")
st.markdown("🔬 Advanced Cell Counter with SAM | Robust segmentation for variable images")
