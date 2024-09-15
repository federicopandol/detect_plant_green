import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the latest image
image_path = 'fotos3.jpeg'
imagen = cv2.imread(image_path)

# Convert the image from BGR to RGB
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Convert to HSV color space
imagen_hsv = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2HSV)

# Refine the green color range for vegetation/trees
verde_bajo = np.array([35, 50, 50])
verde_alto = np.array([85, 255, 255])

# Create a mask for the green areas
mascara = cv2.inRange(imagen_hsv, verde_bajo, verde_alto)

# Use morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
mascara_limpia = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel)

# Apply Canny edge detection to the original image (without mask) for more detail
edges = cv2.Canny(imagen_rgb, 50, 150)

# Find contours from the cleaned mask (after edges)
contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter the contours based on a larger minimum area to focus on bigger objects
min_area = 1000  # A reasonable size for trees in the image (can adjust further)
filtered_contours = [cnt for cnt in contornos if cv2.contourArea(cnt) > min_area]

# Draw bounding boxes around filtered trees
imagen_bboxes = imagen_rgb.copy()
for cnt in filtered_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(imagen_bboxes, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the original and result
plt.figure(figsize=(15, 10))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(imagen_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Tree Detection with Improved Contour Strategy')
plt.imshow(imagen_bboxes)
plt.axis('off')

plt.show()