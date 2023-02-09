'''
main
transform process: image to torch tensor: torchvision.transforms.ToTensor()
normalize the value into 0-1: torchvision.transforms.normalize()

shuffle=True -- don't have the same sequence when training data comes in
in testing data, shuffle could be false, since only test the accuracy

train_loader is the iterator
trainer sents model to trainer.
'''