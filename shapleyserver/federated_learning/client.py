import os
import copy
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

class ClientBase(object):
    def __init__(self, id, args, net_train, train_set, test_set=None):
        self.id = id
        self.args = args
        self.device = self.args.device
        self.model_train = copy.deepcopy(net_train).to(self.device) 
                
        self.local_data_train = train_set
        self.num_local_data_train = len(self.local_data_train)
        self.batch_train = args.client_batch_train
        self.local_trainloader = DataLoader(self.local_data_train, batch_size=self.batch_train, shuffle=False)
        
        if test_set is not None:
            self.local_data_test = test_set
            self.num_local_data_test = len(self.local_data_test)
            self.local_testloader = DataLoader(self.local_data_test, batch_size=2*self.batch_train, shuffle=False, num_workers=args.n_workers)       
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = None
        self.num_local_epochs = args.client_epoch_train # used by FedAvg or local training
        # self.num_local_updates = args.client_num_updates
        self.save_path = os.path.join(self.args.save_path, 'client_{}'.format(self.id))
    
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

    def local_training(self,  trainloader, net_train, optimizer, criterion, do_label_flip=False, label_flip_prob=0.5, logger=None):
        r""" used for baseline method such as FedAvg
        """
        net_train.train()
        for i in range(self.num_local_epochs):
            train_loss, n_samples = 0.0, 0
            for x, y in trainloader: 
                x, y = x.to(self.device), y.to(self.device)

                if do_label_flip:
                    # randomly flip labels with probability label_flip_prob
                    mask = torch.rand(y.size()) < label_flip_prob
                    random_label = torch.randint(y.min(), y.max() + 1, y[mask].size()).to(self.device)
                    # add random label to the original label
                    # print(y)
                    y[mask] = (y[mask] + random_label + 1) % (y.max() - y.min() + 1)
                    # print(y)
                    # print()
                    # print(y.size(), random_label.size())
                    
                output = net_train(x)
                loss = criterion(output,y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * y.size(0)
                n_samples += y.size(0)  
            
            epoch_loss = train_loss/n_samples
            if logger is not None:
                logger.info('Client {:2d}: local epoch {:2d} train loss: {:.8f}'.format(self.id, i, epoch_loss))
