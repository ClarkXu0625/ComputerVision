from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

image_names = ["image1.jpeg","image2.jpeg","image3.jpeg",
                "image4_side.jpeg","image5_crowd.jpeg",
                "image6_crowd.webp","image7_crowd.jpg"]


### question 1
# hindered vs not hindered, both false
result1 = DeepFace.verify(img1_path="image1.jpeg", img2_path="image2.jpeg")
result2 = DeepFace.verify(img1_path="image1.jpeg", img2_path="image3.jpeg")

# not hindered front face
result3 = DeepFace.verify(img1_path="image2.jpeg", img2_path="image3.jpeg")

# front vs side
result4 = DeepFace.verify(img1_path="image2.jpeg", img2_path="image4_side.jpeg")
result5 = DeepFace.verify(img1_path="image3.jpeg", img2_path="image4_side.jpeg")

# single vs crowd
result6 = DeepFace.verify(img1_path="image3.jpeg", img2_path="image5_crowd.jpeg")
result7 = DeepFace.verify(img1_path="image3.jpeg", img2_path="image6_side.webp")
result8 = DeepFace.verify(img1_path="image3.jpeg", img2_path="image7_side.jpg")

results = [result1,result2,result3,result4,result5,result6,result7,result8]
print(results[1])

### Question 2
def analyze(image):
    objs = DeepFace.analyze(img_path=image,
                            actions = ['age', 'gender', 'race', 'emotion'])
    print(objs)

analyze("image4_side.jpeg")


### question 3
backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe'
]
# define a function to print the detected area of the image
def print_face(image):
    #face detection and alignment
    face_objs = DeepFace.extract_faces(img_path = image, 
            target_size = (224, 224), 
            detector_backend = backends[4]
    )
    face_img = face_objs[0]["facial_area"]
    x = face_img["x"]
    y = face_img["y"]
    w = face_img["w"]
    h = face_img["h"]

    # Load the image file
    img_path = image
    img = Image.open(img)

    # Extract the area from the image
    area = img.crop((x, y, x+w, y+h))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # Apply the transforms to the image and convert it to a tensor
    img_tensor = transform(area)

    # Print the shape of the tensor
    face_img_display = np.transpose(img_tensor.numpy(), (1, 2, 0))
    plt.imshow(face_img_display)
    plt.show()

print_face(image_names[4])