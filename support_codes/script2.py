from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from scipy.io import loadmat 
import matplotlib.pyplot as plt
# Load the image
image_path = 'image_0.tif'
image = Image.open(image_path)

# Convert the image to a NumPy array
image_array = np.array(image)
filters = loadmat('filters.mat')['filters']  
result = convolve2d(image_array, filters[:,:,1], mode='same', boundary='wrap', fillvalue=0)
# Normalize the result
result = (result - result.min()) / (result.max() - result.min())  # Normalize to [0, 1]
result = (result * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Convolved Image
plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')
plt.title('Convolved Image')
plt.axis('off')

plt.show()