'''
x used in forward function is a tensor
64-image batch size means x has the size of 64*1*28*28
x -> y
y is 64*10, with the possibility that belongs to each of the 10 classes.

need to ensure the same input and output size
'''

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        