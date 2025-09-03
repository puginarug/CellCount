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
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# Title and description
st.title("🔬 Batch Cell Counter")
st.markdown("""
Upload multiple microscopy images and get cell counts for all of them in a single CSV file.
Adjust the detection parameters below, then process all images at once.
""")

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("1️⃣ Upload Images")
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
    
    st.header("2️⃣ Detection Settings")
    st.markdown("*These settings will apply to all images*")
    
    blur_size = st.slider(
        "Blur Amount (reduces noise)", 
        min_value=1, 
        max_value=21, 
        value=5, 
        step=2,
        help="Higher values remove more noise but might merge nearby cells"
    )
    
    threshold_value = st.slider(
        "Threshold Level", 
        min_value=0, 
        max_value=255, 
        value=127,
        help="Adjust to separate cells from background"
    )
    
    min_area = st.slider(
        "Minimum Cell Size (pixels)", 
        min_value=10, 
        max_value=500, 
        value=50,
        help="Ignore objects smaller than this"
    )
    
    max_area = st.slider(
        "Maximum Cell Size (pixels)", 
        min_value=100, 
        max_value=5000, 
        value=2000,
        help="Ignore objects larger than this"
    )
    
    # Add preview option
    if uploaded_files:
        st.header("3️⃣ Preview Settings")
        preview_file = st.selectbox(
            "Select an image to preview settings:", 
            options=[f.name for f in uploaded_files],
            help="Preview how settings work on one image before processing all"
        )
        
        if st.button("Preview Selected Image"):
            # Find the selected file
            selected_file = next(f for f in uploaded_files if f.name == preview_file)
            
            # Process the preview
            image = Image.open(selected_file)
            img_array = np.array(image)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply processing
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
            _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours
            valid_contours = []
            result_img = img_array.copy()
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    valid_contours.append(contour)
                    cv2.drawContours(result_img, [contour], -1, (0, 255, 0), 2)
            
            # Display preview
            st.image(result_img, caption=f"Preview: {len(valid_contours)} cells detected", use_column_width=True)
            
            with st.expander("View binary threshold"):
                st.image(binary, caption="Binary image", use_column_width=True)

with col2:
    st.header("4️⃣ Process & Results")
    
    if uploaded_files:
        if st.button("🚀 Process All Images", type="primary", use_container_width=True):
            # Clear previous results
            st.session_state.results = []
            st.session_state.processed_files = set()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                try:
                    # Read image
                    image = Image.open(uploaded_file)
                    img_array = np.array(image)
                    
                    # Convert to grayscale
                    if len(img_array.shape) == 3:
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = img_array
                    
                    # Apply processing
                    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
                    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
                    
                    # Find contours
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Filter contours by area
                    valid_contours = []
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if min_area <= area <= max_area:
                            valid_contours.append(contour)
                    
                    # Store result
                    cell_count = len(valid_contours)
                    st.session_state.results.append({
                        'Filename': uploaded_file.name,
                        'Cell_Count': cell_count
                    })
                    st.session_state.processed_files.add(uploaded_file.name)
                    
                except Exception as e:
                    st.session_state.results.append({
                        'Filename': uploaded_file.name,
                        'Cell_Count': f"Error: {str(e)}"
                    })
                
                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ Processing complete!")
    
    # Display results if available
    if st.session_state.results:
        st.success(f"Successfully processed {len(st.session_state.results)} images")
        
        # Create DataFrame
        df = pd.DataFrame(st.session_state.results)
        
        # Display summary statistics
        valid_counts = df[df['Cell_Count'].apply(lambda x: isinstance(x, int))]
        if not valid_counts.empty:
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Total Images", len(df))
            with col_stat2:
                st.metric("Average Cells/Image", f"{valid_counts['Cell_Count'].mean():.1f}")
            with col_stat3:
                st.metric("Total Cells", valid_counts['Cell_Count'].sum())
        
        # Display results table
        st.subheader("Results Table")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Download button for CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="cell_counts.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Option to clear results and start over
        if st.button("🔄 Clear Results & Start New Batch", use_container_width=True):
            st.session_state.results = []
            st.session_state.processed_files = set()
            st.rerun()
    else:
        st.info("👈 Upload images and click 'Process All Images' to get started")

# Instructions
with st.expander("📖 How to Use This Tool"):
    st.markdown("""
    ### Quick Start:
    1. **Upload all your images** at once using the file uploader
    2. **Adjust detection parameters** if needed (use preview to test)
    3. **Click 'Process All Images'** to count cells in all images
    4. **Download the CSV file** with results
    
    ### The CSV file contains:
    - **Filename**: The name of each image file
    - **Cell_Count**: Number of cells detected in that image
    
    ### Tips for Best Results:
    - Use the **preview feature** to test settings on one image before processing all
    - Ensure images have good contrast between cells and background
    - All images will be processed with the same settings
    - Process similar images together for best results
    
    ### Typical Settings:
    - **Fluorescence microscopy**: Lower threshold (30-80)
    - **Bright field**: Higher threshold (100-180)
    - **High magnification**: Smaller cell sizes
    - **Low magnification**: Larger cell sizes
    """)

# Footer
st.markdown("---")
st.markdown("🔬 Batch Cell Counter | Made for easy lab use")