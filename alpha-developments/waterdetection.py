import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt

# Input Image Path
input_image = './main2.png'
img = cv2.imread(input_image)

# Convert the image from BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define color boundaries for water (expanded range)
lower_water = np.array([90, 50, 50])  # Lower hue adjusted for darker blues
upper_water = np.array([140, 255, 255])  # Higher hue to capture lighter blues

# Create a mask for water detection
mask_water = cv2.inRange(hsv, lower_water, upper_water)

# Morphological operations to close small gaps in water areas
water_kernel_threshold = 12
water_kernel = np.ones((water_kernel_threshold, water_kernel_threshold), np.uint8)
closed_water_mask = cv2.morphologyEx(mask_water, cv2.MORPH_CLOSE, water_kernel)

# Find contours of the water mask
contours, _ = cv2.findContours(closed_water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define the minimum area threshold
min_area_threshold = 500  # Adjust this value based on your requirements

# Create a new mask to store filtered water areas
filtered_water_mask = np.zeros_like(closed_water_mask)

# Filter contours by area and fill them in the new mask
for cnt in contours:
    if cv2.contourArea(cnt) >= min_area_threshold:
        cv2.drawContours(filtered_water_mask, [cnt], -1, 255, thickness=cv2.FILLED)

# Convert the filtered mask to uint8 type for saving
water_mask_uint8 = (filtered_water_mask > 0).astype(np.uint8) * 255  # Convert to binary image

# Create GeoTIFF from the water mask
def create_geotiff(output_mask, img_shape, output_path):
    transform = from_origin(0, 0, 1, 1)  # Adjust the transform as necessary
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=img_shape[0],
        width=img_shape[1],
        count=1,
        dtype='uint8',
        crs='+proj=latlong',  # Example CRS, adjust as needed
        transform=transform,
    ) as dst:
        dst.write(output_mask, 1)  # Write the mask to GeoTIFF

# Save the GeoTIFF
output_tiff_path = 'water_map.tiff'  # Path for GeoTIFF output
create_geotiff(water_mask_uint8, img.shape[:2], output_tiff_path)

print(f"Water map saved as {output_tiff_path}")

# Plot original image and water mask
def plot_results(original_image, water_mask):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Water mask
    axes[1].imshow(water_mask, cmap='Blues')
    axes[1].set_title('Water Mask')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# Plot the results
plot_results(img, water_mask_uint8)
