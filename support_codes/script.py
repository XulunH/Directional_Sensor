from keras.datasets import fashion_mnist
from PIL import Image
# Load the Fashion MNIST dataset
(train_images, train_labels), (_, _) = fashion_mnist.load_data()

# Select the first image
image = train_images[160]


# Convert the image array into a PIL Image object
pil_img = Image.fromarray(image)

# Save the image in TIF format
pil_img.save('fashion_mnist_sample2.tif')
