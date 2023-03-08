import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class FaceVerificationDataset(Dataset):
    def __init__(self, root_dir, label_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label_dir = label_dir
        # Load the image pairs and labels from a file
        self.image_pairs, self.labels = self.load_pairs_file()

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        # Load the two images and label at the given index
        img1_path, img2_path = self.image_pairs[index]
        img1 = Image.open(os.path.join(self.root_dir, img1_path)).convert('RGB')
        img2 = Image.open(os.path.join(self.root_dir, img2_path)).convert('RGB')
        label = self.labels[index]

        # Apply the transformation to both images if specified
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # Convert the label to a tensor
        label = torch.tensor(label, dtype=torch.float32)

        return img1, img2, label

    def load_pairs_file(self):
        # Load the image pairs and labels from a file
        pairs_file = os.path.join(self.root_dir, self.label_dir)
        image_pairs = []
        labels = []
        
        with open(pairs_file, "r") as train_file:
            size =int(train_file.readline())    # dataset labels number of pairs at the first line
            
            for i in range(size):
                text = train_file.readline()
                temp = text.replace("\n", "").split("\t")
                path1 = temp[0]+"/"+temp[0]+"_"+ '{:0>4s}'.format(temp[1])+".jpg"
                path2 = temp[0]+"/"+temp[0]+"_"+ '{:0>4s}'.format(temp[2])+".jpg"
                image_pairs.append((path1, path2))
                labels.append(0)
                i+=1
            for i in range(size):
                text = train_file.readline()
                temp = text.replace("\n", "").split("\t")
                path1 = temp[0]+"/"+temp[0]+"_"+ '{:0>4s}'.format(temp[1])+".jpg"
                path2 = temp[2]+"/"+temp[2]+"_"+ '{:0>4s}'.format(temp[3])+".jpg"
                image_pairs.append((path1, path2))
                labels.append(1)
                i+=1
            
        return image_pairs, labels


#f = FaceVerificationDataset("./data/lfw","pairsDevToy.txt")
