'''
epochs: number of training, steps of model converge. 
    look at the loss function
    set max epochs
learning rate (ls)
batch size (bs)

'''

from train import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="mnist classification")
    parser.add_argument('--epochs', type=int, defualt=10, help="training epochs")


def main():
    args = parse_args()
    
