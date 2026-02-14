import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from scipy.ndimage import gaussian_filter
import torch
import requests
from io import BytesIO
import cv2
import warnings
warnings.filterwarnings('ignore')
from transformers import DPTImageProcessor, DPTForDepthEstimation

# Global model cache
segformer_processor = None
segformer_model = None
dpt_processor = None
dpt_model = None

def load_models():
    global segformer_processor, segformer_model, dpt_processor, dpt_model
    
    if segformer_processor is None:
        print("Loading Segformer model...")
        segformer_processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        segformer_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
    
    if dpt_processor is None:
        print("Loading DPT model...")
        dpt_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    
    return segformer_processor, segformer_model, dpt_processor, dpt_model

def preprocess_image(image, target_size=512):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to square
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return image

def segment_human(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=(512, 512),
        mode="bilinear",
        align_corners=False
    )
    
    pred_seg = upsampled.argmax(dim=1)[0].cpu().numpy()
    human_mask = (pred_seg == 12).astype(np.uint8) * 255
    return human_mask

def apply_background_blur(image, mask, sigma=15):
    img_array = np.array(image).astype(np.float32)
    mask_normalized = mask.astype(np.float32) / 255.0
    
    # Smooth edges
    mask_smooth = gaussian_filter(mask_normalized, sigma=2)
    mask_smooth = np.clip(mask_smooth, 0, 1)
    
    # Blur entire image
    blurred_array = np.zeros_like(img_array)
    for i in range(3):
        blurred_array[:, :, i] = gaussian_filter(img_array[:, :, i], sigma=sigma)
    
    # Blend
    mask_3d = np.stack([mask_smooth] * 3, axis=2)
    result = (img_array * mask_3d + blurred_array * (1 - mask_3d)).astype(np.uint8)
    return Image.fromarray(result)

def estimate_depth(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(512, 512),
        mode="bicubic",
        align_corners=False,
    )
    
    depth_map = prediction.squeeze().cpu().numpy()
    
    # Normalize to 0-15 and INVERT (high=far for proper blur)
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    normalized = (depth_map - depth_min) / (depth_max - depth_min)
    normalized = 1.0 - normalized  # INVERT: high = far
    depth_normalized = normalized * 15.0
    return depth_normalized

def apply_lens_blur(image, depth_map, max_sigma=15):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR).astype(np.float32)
    
    # Create blur pyramid
    num_levels = 10
    blur_pyramid = []
    
    for i in range(num_levels):
        sigma = (i / (num_levels - 1)) * max_sigma
        if sigma < 0.5:
            blur_pyramid.append(img_cv.copy())
        else:
            ksize = int(2 * np.ceil(3 * sigma) + 1)
            if ksize % 2 == 0:
                ksize += 1
            blurred = cv2.GaussianBlur(img_cv, (ksize, ksize), sigma)
            blur_pyramid.append(blurred)
    
    # Apply variable blur based on depth
    depth_norm = depth_map / 15.0
    output = np.zeros_like(img_cv)
    
    depth_scaled = depth_norm * (num_levels - 1)
    level_low = np.floor(depth_scaled).astype(np.int32)
    level_high = np.ceil(depth_scaled).astype(np.int32)
    level_low = np.clip(level_low, 0, num_levels - 1)
    level_high = np.clip(level_high, 0, num_levels - 1)
    
    weight = depth_scaled - level_low
    weight = np.expand_dims(weight, axis=2)
    
    for y in range(img_cv.shape[0]):
        for x in range(img_cv.shape[1]):
            ll = level_low[y, x]
            lh = level_high[y, x]
            w = weight[y, x, 0]
            
            if ll == lh:
                output[y, x] = blur_pyramid[ll][y, x]
            else:
                output[y, x] = (1 - w) * blur_pyramid[ll][y, x] + w * blur_pyramid[lh][y, x]
    
    output = np.clip(output, 0, 255).astype(np.uint8)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return Image.fromarray(output_rgb)

def process_gaussian_blur(image, sigma):
    if image is None:
        return None, "Please upload an image first!"
    
    try:
        # Load models
        seg_proc, seg_model, _, _ = load_models()
        
        # Preprocess
        img = preprocess_image(image, 512)
        
        # Segment human
        mask = segment_human(img, seg_proc, seg_model)
        
        # Apply blur
        result = apply_background_blur(img, mask, sigma)
        
        return result, f"Gaussian blur (σ={sigma}) applied successfully!"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def process_lens_blur(image, max_sigma):
    if image is None:
        return None, None, "Please upload an image first!"
    
    try:
        # Load models
        _, _, dpt_proc, dpt_model = load_models()
        
        # Preprocess
        img = preprocess_image(image, 512)
        
        # Estimate depth
        depth = estimate_depth(img, dpt_proc, dpt_model)
        
        # Apply lens blur
        result = apply_lens_blur(img, depth, max_sigma)
        
        # Create depth visualization
        depth_vis = ((depth / 15.0) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        depth_img = Image.fromarray(depth_colored)
        
        return result, depth_img, f"Lens blur (σ_max={max_sigma}) applied successfully!"
    
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="GaussBlur App", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # GaussBlur App
    ### Professional Image Blur Effects using AI
    
    Two powerful blur techniques:
    1. **Gaussian Background Blur** - Zoom-style video conferencing effect
    2. **Depth-Based Lens Blur** - Realistic camera depth of field
    
    ---
    """)
    
    with gr.Tabs():
        # Tab 1: Gaussian Background Blur
        with gr.Tab("📹 Gaussian Background Blur"):
            gr.Markdown("""
            ### Zoom-Style Background Blur
            Segments the human and blurs only the background with uniform Gaussian blur.
            Perfect for video conferencing effects!
            """)
            
            with gr.Row():
                with gr.Column():
                    gaussian_input = gr.Image(label="Upload Image", type="numpy")
                    gaussian_sigma = gr.Slider(
                        minimum=0, 
                        maximum=30, 
                        value=15, 
                        step=1,
                        label="Blur Strength (σ)",
                        info="Higher values = more blur"
                    )
                    gaussian_btn = gr.Button("Apply Gaussian Blur", variant="primary")
                
                with gr.Column():
                    gaussian_output = gr.Image(label="Result")
                    gaussian_status = gr.Textbox(label="Status", lines=2)
            
            gr.Markdown("""
            **How it works:**
            1. AI segments human from background using Segformer
            2. Applies uniform Gaussian blur (σ=15) to background only
            3. Keeps human sharp with smooth edge transitions
            """)
        
        # Tab 2: Depth-Based Lens Blur
        with gr.Tab("📸 Depth-Based Lens Blur"):
            gr.Markdown("""
            ### Realistic Camera Depth of Field
            Estimates depth and applies variable blur based on distance.
            Close objects stay sharp, far objects get blurred - just like a real camera!
            """)
            
            with gr.Row():
                with gr.Column():
                    lens_input = gr.Image(label="Upload Image", type="numpy")
                    lens_sigma = gr.Slider(
                        minimum=0,
                        maximum=25,
                        value=15,
                        step=1,
                        label="Maximum Blur (σ_max)",
                        info="Blur for farthest objects"
                    )
                    lens_btn = gr.Button("Apply Lens Blur", variant="primary")
                
                with gr.Column():
                    lens_output = gr.Image(label="Blurred Result")
                    lens_depth = gr.Image(label="Depth Map (Purple=Close, Yellow=Far)")
                    lens_status = gr.Textbox(label="Status", lines=2)
    
    # Connect buttons to functions
    gaussian_btn.click(
        fn=process_gaussian_blur,
        inputs=[gaussian_input, gaussian_sigma],
        outputs=[gaussian_output, gaussian_status]
    )
    
    lens_btn.click(
        fn=process_lens_blur,
        inputs=[lens_input, lens_sigma],
        outputs=[lens_output, lens_depth, lens_status]
    )

# Launch app
if __name__ == "__main__":
    print("Starting Blur Effects Studio...")
    print("Loading AI models (this may take a moment)...")
    load_models()
    print("Models loaded! Launching app...")
    demo.launch(share=True)