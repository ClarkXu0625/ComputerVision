import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


image = Image.open("sample.jpg")

# define how to resize, horizontally flip, and turn image to gray
resize = transforms.Resize((1000,1000))
hflip = transforms.RandomHorizontalFlip(p=1)
gray = transforms.Grayscale()

# Apply the transforms to the image
resized_image = resize(image)
flipped_image = hflip(resized_image)
gray_image = gray(flipped_image)

# Convert the images to numpy arrays
resized_array = np.array(resized_image)
flipped_array = np.array(flipped_image)
gray_array = np.array(gray_image)

# Convert gray image to three dimension (1000,1000,1)
# Then replicate the image in first two dimensions, result as (1000,1000,3)
color_image = np.expand_dims(gray_array, axis=-1)
rgb_image = np.repeat(color_image, 3, axis=-1)

# Concatenate the images into a single row
concatenated_image = np.concatenate((resized_array, flipped_array, rgb_image), axis=1)

# Convert the concatenated image back to a PIL image
concatenated_pil_image = Image.fromarray(concatenated_image)

# Save the concatenated image
concatenated_pil_image.save("concatenated_image.jpg")