'''Notes from class: main
transform process: image to torch tensor: torchvision.transforms.ToTensor()
normalize the value into 0-1: torchvision.transforms.normalize()

shuffle=True -- don't have the same sequence when training data comes in
in testing data, shuffle could be false, since only test the accuracy

train_loader is the iterator
trainer sents model to trainer.
'''
import argparse

import torch
import torchvision

from model import Net
from train import Trainer
from evaluation import evaluation   # evaluation class to measure prediction accuracy
from inference import inference     # inference class to predict class from given image



def parse_args():
    parser = argparse.ArgumentParser(description='mnist classification')
    parser.add_argument('--epochs', type=int, default=10, help="training epochs")
    parser.add_argument('--lr', type=float, default=1e-1, help="learning rate")
    parser.add_argument('--bs', type=int, default=64, help="batch size")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # model
    model = Net()

    # datasets
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=False,
    )

    # trainer
    trainer = Trainer(model=model)

    # model training
    trainer.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, save_dir="./save/")

    # model evaluation
    evaluator = evaluation(model=model)
    print(evaluator.eval(test_loader=test_loader))

    # model inference
    sample = torch.rand(1,28,28)  # complete the sample here
    inferencer = inference(model=model)
    inferencer.infer(sample=sample)
    #img = np.transpose(sample.numpy(),(1,2,0))
    #plt.imshow(img)
    #plt.show()

    return


if __name__ == "__main__":
    main()
