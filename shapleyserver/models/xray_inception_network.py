import torch
import torch.nn as nn
import torch.nn.functional as F

#from models.resnet import resnet50
from . inception import Inception3
from .. opts import opt

class inception_network(nn.Module):
    '''
                end to end net
    '''

    def __init__(self):

        super(inception_network, self).__init__()
        if opt.dataset_type == 'messidor' or opt.dataset_type=='cell' or opt.dataset_type == 'alzhm':
            num_classes = 4
        elif opt.dataset_type == 'AML':
            num_classes = 3
        else:
            num_classes = 2
        self.inception = Inception3(denoise=None, num_classes=4)

    def forward(self, input, defense=False):
        x = self.inception(input, defense)
        return x
