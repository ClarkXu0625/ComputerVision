import argparse

import torch
import torchvision

from model import Net, sphere20a
from train import Trainer
from evaluation import evaluation   # evaluation class to measure prediction accuracy
from inference import inference     # inference class to predict class from given image
from dataset import FaceVerificationDataset

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='sphereface')
    parser.add_argument('--epochs', type=int, default=10, help="training epochs")
    parser.add_argument('--lr', type=float, default=1e-1, help="learning rate")
    parser.add_argument('--bs', type=int, default=64, help="batch size")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # model
    model = sphere20a()

    # datasets
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = FaceVerificationDataset("./data/lfw", "pairsDevTrain.txt")

    test_loader = FaceVerificationDataset("./data/lfw", "pairsDevTest.txt")

    # trainer
    trainer = Trainer(model=model)

    # model training
    trainer.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, save_dir="./save/")

    ### model evaluation
    evaluator = evaluation(model=model)
    print("Accuracy for test images prediction is: "+ str(evaluator.eval(test_loader=test_loader)) + "%")

    ### model inference, print predicted class from given sample image tensor   
    # Get a random image from the data loader
    images, label = next(iter(test_loader)) 
    index = torch.randint(0, images.shape[0], size=(1,))    # Generate a random index   
    selected_tensor = images[index] # Select a single tensor from the larger tensor using the index
    sample = selected_tensor.reshape(1, 28, 28)

    # perform the inference task
    inferencer = inference(model=model)
    print("The predicted class from given image is: "+ str(inferencer.infer(sample=sample)))

    # print the random image
    image_np = np.squeeze(sample.numpy())
    plt.imshow(image_np, cmap='gray')
    plt.show()

    return


if __name__ == "__main__":
    main()
