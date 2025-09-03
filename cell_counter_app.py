import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io

# Page configuration
st.set_page_config(
    page_title="Batch Cell Counter",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []

# Title and description
st.title("🔬 Batch Cell Counter with Visual Validation")
st.markdown("""
Upload multiple microscopy images and get cell counts with visual verification.
You can review the segmentation quality for each image before downloading results.
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
        
        # Display list of uploaded files
        with st.expander("View uploaded files"):
            for file in uploaded_files:
                st.text(f"📄 {file.name}")
    
    st.header("⚙️ Detection Settings")
    st.markdown("*Applied to all images*")
    
    blur_size = st.slider(
        "Blur Amount", 
        min_value=1, 
        max_value=21, 
        value=5, 
        step=2,
        help="Reduces noise"
    )
    
    threshold_value = st.slider(
        "Threshold Level", 
        min_value=0, 
        max_value=255, 
        value=127,
        help="Separates cells from background"
    )
    
    min_area = st.slider(
        "Min Cell Size (px)", 
        min_value=10, 
        max_value=500, 
        value=50,
        help="Ignore smaller objects"
    )
    
    max_area = st.slider(
        "Max Cell Size (px)", 
        min_value=100, 
        max_value=5000, 
        value=2000,
        help="Ignore larger objects"
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
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    try:
                        # Read image
                        image = Image.open(uploaded_file)
                        img_array = np.array(image)
                        original_img = img_array.copy()
                        
                        # Convert to grayscale
                        if len(img_array.shape) == 3:
                            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        else:
                            gray = img_array
                            # Convert grayscale to RGB for display
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                            original_img = img_array.copy()
                        
                        # Apply processing
                        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
                        _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
                        
                        # Find contours
                        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Filter contours and create result image
                        valid_contours = []
                        result_img = img_array.copy()
                        
                        for i, contour in enumerate(contours):
                            area = cv2.contourArea(contour)
                            if min_area <= area <= max_area:
                                valid_contours.append(contour)
                                
                                # Draw contour
                                if show_contours:
                                    cv2.drawContours(result_img, [contour], -1, (0, 255, 0), 2)
                                
                                # Get center
                                M = cv2.moments(contour)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    
                                    # Draw center
                                    if show_centers:
                                        cv2.circle(result_img, (cx, cy), 4, (255, 0, 0), -1)
                                    
                                    # Draw number
                                    if show_numbers:
                                        cv2.putText(result_img, str(len(valid_contours)), 
                                                  (cx+7, cy-7), cv2.FONT_HERSHEY_SIMPLEX, 
                                                  0.6, (255, 255, 0), 2)
                        
                        # Store result
                        cell_count = len(valid_contours)
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
            # Download button for CSV (if results available)
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
    
    # Display results if available
    if st.session_state.results:
        # Summary statistics
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
        
        # Display results table
        st.subheader("📊 Results Summary")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Visual results section
        st.markdown("---")
        st.subheader("🖼️ Visual Verification")
        st.markdown("Review the segmentation quality for each image below:")
        
        # Image viewer with navigation
        if st.session_state.processed_images:
            # Image selector
            image_names = [img['filename'] for img in st.session_state.processed_images]
            selected_image = st.selectbox(
                "Select image to review:",
                options=range(len(image_names)),
                format_func=lambda x: f"{image_names[x]} ({st.session_state.processed_images[x]['count']} cells)"
            )
            
            # Display selected image pair
            img_data = st.session_state.processed_images[selected_image]
            
            # Create two columns for side-by-side display
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                st.markdown("**Original Image**")
                st.image(img_data['original'], use_column_width=True)
            
            with img_col2:
                st.markdown(f"**Segmentation Result** ({img_data['count']} cells)")
                st.image(img_data['segmented'], use_column_width=True)
            
            # Option to view binary image
            with st.expander("View threshold image (for troubleshooting)"):
                st.image(img_data['binary'], caption="Binary threshold image", use_column_width=True)
            
            # Navigation buttons
            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            with nav_col1:
                if selected_image > 0:
                    if st.button("← Previous", use_container_width=True):
                        st.session_state.selected_image = selected_image - 1
                        st.rerun()
            
            with nav_col2:
                st.markdown(f"<center>Image {selected_image + 1} of {len(image_names)}</center>", unsafe_allow_html=True)
            
            with nav_col3:
                if selected_image < len(image_names) - 1:
                    if st.button("Next →", use_container_width=True):
                        st.session_state.selected_image = selected_image + 1
                        st.rerun()
        
        # Clear results button
        st.markdown("---")
        if st.button("🔄 Clear All & Start New Batch", use_container_width=True):
            st.session_state.results = []
            st.session_state.processed_images = []
            st.rerun()
    else:
        st.info("👈 Upload images and click 'Process All Images' to get started")

# Instructions
with st.expander("📖 How to Use This Tool"):
    st.markdown("""
    ### Quick Start:
    1. **Upload all your images** at once using the file uploader
    2. **Adjust detection parameters** if needed
    3. **Click 'Process All Images'** to count cells in all images
    4. **Review the visual results** to verify segmentation quality
    5. **Download the CSV file** with results
    
    ### Visual Verification:
    - After processing, you can review each image side-by-side with its segmentation
    - Green outlines = detected cell boundaries
    - Blue dots = cell centers
    - Yellow numbers = cell count
    - Use the dropdown or navigation buttons to review all images
    
    ### The CSV file contains:
    - **Filename**: The name of each image file
    - **Cell_Count**: Number of cells detected in that image
    
    ### Tips for Best Results:
    - Adjust threshold until cells are clearly separated in the binary view
    - Use blur to reduce noise in grainy images
    - Set min/max size based on your typical cell sizes
    - Check the visual results to ensure accurate detection
    """)

# Footer
st.markdown("---")
st.markdown("🔬 Batch Cell Counter with Visual Validation | Easy lab analysis tool")