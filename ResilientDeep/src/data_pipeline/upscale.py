import cv2
import numpy as np

def simulate_compression(image, quality=15):
    """
    Applies heavy JPEG compression to destroy high-frequency artifacts.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def apply_bicubic(image, scale_factor=2):
    """
    Standard Bicubic upscaling.
    """
    h, w = image.shape[:2]
    return cv2.resize(image, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)

def create_attack_image(image_path, save_path):
    """
    Executes the compression -> upscaling pipeline.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False
        
    compressed = simulate_compression(img)
    upscaled = apply_bicubic(compressed)
    
    cv2.imwrite(save_path, upscaled)
    return True

# Note: Full ESRGAN via BasicSR requires downloading a heavy .pth weights file.
# We are starting with Bicubic to validate the pipeline logic first.
