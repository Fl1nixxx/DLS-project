import numpy as np
import cv2

import math

import rasterio
from rasterio.io import MemoryFile

def count_building_area(mask, pixel_area, noise_threshold=10):
    binary_mask = np.array(mask)
    if binary_mask.max() == 1:
        binary_mask = (binary_mask * 255).astype(np.uint8)
    else:
        binary_mask = (binary_mask > 127).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    buildings_report = []
    noise_count = 0

    for cnt in contours:
        pixel_count = cv2.contourArea(cnt)
        area_sqm = pixel_count * pixel_area

        if area_sqm < noise_threshold:
            noise_count += 1
            continue

        area_rounded = round(area_sqm, 2)
        buildings_report.append(area_rounded)
        
    total_area = sum(buildings_report)
    
    return total_area

def define_pixel_area(uploaded_file):
    try:
        uploaded_file.seek(0)
        with MemoryFile(uploaded_file.read()) as memfile:
            with memfile.open() as src:
                crs = src.crs
                x, y = src.res
                
                if src.bounds.left == 0.0 and src.bounds.bottom == 0.0:
                    return None 
                
                if crs and crs.is_projected:
                    pixel_area = x * y
                else:
                    lat = src.bounds.bottom + (src.bounds.top - src.bounds.bottom) / 2
                    meters_per_degree_lat = 111132
                    meters_per_degree_lon = meters_per_degree_lat * math.cos(math.radians(lat))
                    pixel_area = (x * meters_per_degree_lon) * (y * meters_per_degree_lat)
                    
        return pixel_area
    except Exception:
        return None
