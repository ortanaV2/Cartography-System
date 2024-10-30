import numpy as np
import rasterio
from rasterio.transform import from_origin
import detectree as dtr
import matplotlib.pyplot as plt
from rasterio import plot
from PIL import Image

# 1. Bild laden und Segmentierung durchführen
input_image_path = './main2.png'  # Bildpfad anpassen
img = Image.open(input_image_path)

# Verwende DetecTree für die Segmentierung von Baumflächen
y_pred = dtr.Classifier().predict_img(input_image_path)

# 2. GeoTIFF-Erstellung für die Baumkarte
# Definiere die Auflösung und die Ursprungstransformation basierend auf dem Bild
transform = from_origin(0, 0, 1, 1)  # Beispielhafte Transformation, passe an deine Geodaten an
output_tiff_path = 'tree_map.tiff'

with rasterio.open(
    output_tiff_path,
    'w',
    driver='GTiff',
    height=y_pred.shape[0],
    width=y_pred.shape[1],
    count=1,
    dtype='uint8',
    crs='+proj=latlong',  # Beispielhafte CRS, falls Georeferenzierung vorhanden
    transform=transform,
) as dst:
    dst.write(y_pred.astype(np.uint8), 1)  # Schreibe die Baumsegmentierung in das GeoTIFF

# 3. Visualisierung der Baumkarte über dem Originalbild
# Das GeoTIFF und das Originalbild gemeinsam anzeigen
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Originalbild anzeigen
axes[0].imshow(img)
axes[0].set_title('Original Image')

# GeoTIFF-Baumkarte anzeigen
with rasterio.open(output_tiff_path) as src:
    plot.show(src, ax=axes[1], cmap='Greens')  # Grünfärbung für Bäume
    axes[1].set_title('Tree Segmentation Overlay')

plt.tight_layout()
plt.show()
