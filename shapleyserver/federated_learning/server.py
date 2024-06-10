# import os
import os
import numpy as np
import copy
import torch
from torch.utils.data import DataLoader
from federated_learning.utils import (
    init_new_net, 
    add_net_state, 
    add_net_state2,
    add_net_state3
)


class ServerBase(object):
    def __init__(self, args, net_train, clients, test_set, valid_set=None, group_valid_dataset=None):
        # Set up the main attributes
        self.args = args
        self.device = args.device        
        self.global_model = copy.deepcopy(net_train).to(self.device)
        self.clients = clients
        self.num_clients = len(self.clients)
        self.test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=args.n_workers)
        self.valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False, num_workers=args.n_workers)
        # self.save_path = os.path.join(args.save_path, 'server')

        # sensitive group sepecific dataloader
        self.group_valid_loader = []
        if(group_valid_dataset is not None):
            for test_dataset in group_valid_dataset:
                self.group_valid_loader.append(DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=args.n_workers))


    @property
    def net_copy(self):
        return copy.deepcopy(self.global_model).to(self.device)
    
    @property
    def global_model_weights(self):
        return copy.deepcopy(list(self.global_model.parameters()))

    @property
    def global_model_state(self):
        return copy.deepcopy(self.global_model.state_dict())  

    def init_net(self,path=None):
        if path is not None:
            init_weights_state = torch.load(path)
            self.global_model.load_state_dict(init_weights_state['model_state_dict'])
        else:
            self.global_model.apply(init_new_net)
    
    def clients_sel(self, frac=1.0, rng=None):
        r""" Selects a subset of clients from all the clients
        """ 
        if frac<1.0:
            if rng is not None:
                clients_sel = rng.choice(self.clients, self.args.num_clients, replace=False).tolist()
            else:
                clients_sel = np.random.choice(self.clients, self.args.num_clients, replace=False).tolist()
        else:
            clients_sel = self.clients
        return clients_sel

    def get_agg_ratio(self, selected_clients=None):
        r""" Compute coefficients for performing cluster-wise model aggregation

            Default is the same method as FedAvg
        """
        # compute standard FedAvg aggregation coefficients
        if selected_clients is None:
            selected_clients = self.clients
        n_train_list = [client.num_local_data_train for client in selected_clients]
        ratio = [n/sum(n_train_list) for n in n_train_list]

        # put other aggregation ratio methods below
        # ...
        return ratio

    def model_agg(self, parties):
        r""" Implementation of standard model aggregation algorithm of FedAvg
            
            Clients can be a list of actually selected clients or all the clients
        """        
        # compute the aggregated model weights
        global_model_weights_update = add_net_state(parties, ratio=self.get_agg_ratio()) 
        
        # update the state of the (true) global model
        self.global_model.load_state_dict(global_model_weights_update) 
        return global_model_weights_update
    
    def model_agg2(self, nets, selected_clients=None):
        r""" Another implementation of standard model aggregation algorithm of FedAvg
            
            Taking local models as inputs instead of client objects
        """        
        # compute the aggregated model weights
        global_model_weights_update = add_net_state2(nets, ratio=self.get_agg_ratio(selected_clients=selected_clients)) 
        
        # update the state of the (true) global model
        self.global_model.load_state_dict(global_model_weights_update) 
        return global_model_weights_update
    

    def model_agg3(self, server_net, nets, selected_clients=None):
        r""" Another implementation of standard model aggregation algorithm of FedAvg
            
            Taking local models as inputs instead of client objects
        """        
        # compute the aggregated model weights
        global_model_weights_update = add_net_state3(server_net, nets, ratio=self.get_agg_ratio(selected_clients=selected_clients)) 
        
        # update the state of the (true) global model
        self.global_model.load_state_dict(global_model_weights_update) 
        return global_model_weights_update
    

    def model_agg_lazy(self, init_global_model, client_models):
        w_init = copy.deepcopy(init_global_model.state_dict())
        for i, w in enumerate(client_models):
            for key in w.keys():
                # print(key, w_init[key].dtype, w[key].dtype)
                w_init[key] = w_init[key] + w[key]
        self.global_model.load_state_dict(w_init)
        