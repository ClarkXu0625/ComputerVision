import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def loadTrain():
    # Load the data
    train_path = "./lfw"
    val_path = "./lfw"
    train_txt = "pairsDevTrain.txt"
    val_txt = "pairsDevTest.txt"

    train = {}
    ls = []
    with open(train_txt, "r") as train_file:
        size =int(train_file.readline())

        # 
        for i in range(size):
            text = train_file.readline()
            temp = text.replace("\n", "").split("\t")
            pair = ((temp[0], temp[1]), (temp[0], temp[2]))
            train[pair] = True
            i+=1
        for i in range(size):
            text = train_file.readline()
            temp = text.replace("\n", "").split("\t")
            pair = ((temp[0], temp[1]), (temp[2], temp[3]))
            train[pair] = False
            i+=1

    # Define a transformation
    transform = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # # Create a dataset
    train_dataset = datasets.DatasetFolder(train_path, loader=None, extensions=('jpg', 'jpeg', 'png'),
                                transform=transform)
    # val_dataset = datasets.DatasetFolder(val_path, loader=None, extensions=('jpg', 'jpeg', 'png'),
    #                             transform=transform)

    # # Create a DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=True,
    )

    return train_loader


from PIL import Image



