#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from PIL import Image

def convert_images_to_jpg(image_dir="uploaded_images/"):
    """
    Converts all images in a folder to RGB .jpg format with the same filename.
    
    Parameters:
        image_dir (str): Path to the folder containing images.
    """
    # Make sure folder exists
    if not os.path.isdir(image_dir):
        print(f"Directory {image_dir} does not exist.")
        return
    
    for filename in os.listdir(image_dir):
        filepath = os.path.join(image_dir, filename)
        
        # Process only image files (skip hidden/system files)
        if filename.lower().endswith((".png", ".jpeg", ".jpg")):
            try:
                # Open and convert to RGB
                img = Image.open(filepath).convert("RGB")
                
                # Build output filename with .jpg extension
                base, _ = os.path.splitext(filename)
                new_filename = base + ".jpg"
                new_filepath = os.path.join(image_dir, new_filename)
                
                # Save as JPEG
                img.save(new_filepath, "JPEG")
                print(f"✅ Converted {filename} → {new_filename}")
            
            except Exception as e:
                print(f"❌ Could not process {filename}: {e}")

if __name__ == "__main__":
    convert_images_to_jpg("uploaded_images/")
