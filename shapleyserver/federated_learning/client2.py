import os
import copy
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

class ClientBase(object):
    def __init__(self, id, args, net_train, train_set, test_set=None):
        self.id = id
        self.args = args
        #self.device = self.args.device
        #self.model_train = copy.deepcopy(net_train).to(self.device) 
                
        self.local_data_train = train_set
        self.num_local_data_train = len(self.local_data_train)
        #self.batch_train = args.client_batch_train
        #self.local_trainloader = DataLoader(self.local_data_train, batch_size=self.batch_train, shuffle=False)
        
        if test_set is not None:
            self.local_data_test = test_set
            self.num_local_data_test = len(self.local_data_test)
            self.local_testloader = DataLoader(self.local_data_test, batch_size=2*self.batch_train, shuffle=False, num_workers=args.n_workers)       
        
        #self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = None
        #self.num_local_epochs = args.client_epoch_train # used by FedAvg or local training
        # self.num_local_updates = args.client_num_updates
        #self.save_path = os.path.join(self.args.save_path, 'client_{}'.format(self.id))
    
    @property
    def model_weights(self):
        return copy.deepcopy(list(self.model_train.parameters()))

    @property
    def model_state(self):
        return copy.deepcopy(self.model_train.state_dict())  

    def sync_with_server(self, server):
        r""" receive/copy global model of last round to a local cache
            to be used for local training steps
        """
        self.model_train.load_state_dict(server.global_model_state)# download the state of the global model
