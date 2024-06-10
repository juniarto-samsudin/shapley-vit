from .. federated_learning.utils import evaluation, get_aggregated_model
import copy

class Game():
    # a shapley game with n players
    def __init__(self, 
                 clients, 
                 server,
                 init_server_model, 
                 client_models, 
                 client_selection_vector,
                 previous_utility,
                 utility_dim, 
                 server_args
                 ):
        


        self.server = server
        self.clients = clients
        self.init_server_model = init_server_model
        self.client_models = client_models
        self.client_selection_vector = client_selection_vector
        #self._n_all = len(self.clients)
        self._n_all = 3
        self.selected_clients = [i for i in range(self._n_all) if self.client_selection_vector[i]]
        self.n = len(self.selected_clients)
        self.previous_utility = previous_utility
        self.utility_dim = utility_dim
        assert self.utility_dim == 2        
        self.server_args = server_args
        self.utility = []
        for _ in range(self.utility_dim):
            self.utility.append({})

        
        self.compute_default_shapley_value()        


    def compute_default_shapley_value(self):
        self.default_shapley_value = [{client_id: 0 for client_id in range(self._n_all)} for _ in range(self.utility_dim)]
        return 
        # # If the previous utility is non-zero, shapley computation is reduced to only selected clients
        # all_utility_nonzero = True
        # for i in range(self.utility_dim):
        #     if self.previous_utility[i] == 0:
        #         all_utility_nonzero = False
        #         break

        # # Non-trivial Shapley value for non-selected clients
        # if not all_utility_nonzero:
        #     self.default_shapley_value = []
        #     # base performance
        #     base_performance = evaluation(self.server_args, self.init_server_model, self.server.valid_loader)
        #     print(f"Base performance: {base_performance}")
        #     for i in range(self.utility_dim):
        #         self.default_shapley_value.append({})
        #         for client_id in range(self._n_all):
        #             if self.client_selection_vector[client_id]:
        #                 self.default_shapley_value[i][client_id] =  0
        #             else:
        #                 self.default_shapley_value[i][client_id] =  base_performance[i] / self._n_all
        #                 self.previous_utility[i] += self.default_shapley_value[i][client_id]
        # else:
        #     self.default_shapley_value = [{client_id: 0 for client_id in range(self._n_all)} for _ in range(self.utility_dim)]
        
        # # print(f"Default Shapley value: {self.default_shapley_value}")
        # # print(f"Previous utility: {self.previous_utility}")

    def get_default_shapley_value(self):
        return self.default_shapley_value


    def eval_utility(self, coalition):

        coalition = frozenset(coalition)
        # print(f"evaluating utility of {coalition}")

        # null utility
        if len(coalition) == 0:
            return [0 for _ in range(self.utility_dim)]

        # check if utility is already computed
        if coalition in self.utility[0]:
            # print(f"{coalition} already computed")
            return [self.utility[i][coalition] for i in range(self.utility_dim)]

        # get aggregation model
        per_round_aggregated_models = []
        
        participating_indices = [
            j for j in coalition if self.client_selection_vector[j]]
        print("Participating indices: {} ".format(participating_indices)) #[2]
        
        if (len(participating_indices) > 0):
            per_round_aggregated_models.append(
                get_aggregated_model([self.client_models[j] for j in participating_indices],
                                        #self.server.get_agg_ratio(selected_clients=[self.clients[j] for j in participating_indices]))
                                        self.get_agg_ratio(selected_clients=[self.clients[j] for j in participating_indices]))
            )

        #print('per_round_aggregated_models: ', per_round_aggregated_models)


        # compute utility
        self.server.model_agg_lazy(self.init_server_model, per_round_aggregated_models)
        #self.model_agg_lazy(self.init_server_model, per_round_aggregated_models)
        # self.server.global_model.load_state_dict(self.init_server_model.state_dict())
        valid_acc, valid_loss = evaluation(
            self.server_args, self.server.global_model, self.server.valid_loader)
        
        # update
        self.utility[0][coalition] = valid_acc - self.previous_utility[0] # acc
        self.utility[1][coalition] = valid_loss - self.previous_utility[1] # loss

        # print(f"Aggregating models {len(per_round_aggregated_models)} | {coalition} | {[self.utility[i][coalition] for i in range(self.utility_dim)]}")
        
        # return utility of all dimensions as a list
        # if(len(coalition) == 1):
        #     print(coalition, [self.utility[i][coalition] for i in range(self.utility_dim)])
        return [self.utility[i][coalition] for i in range(self.utility_dim)]
        

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
    
    def model_agg_lazy(self, init_global_model, client_models):
        w_init = copy.deepcopy(init_global_model.state_dict())
        for i, w in enumerate(client_models):
            for key in w.keys():
                # print(key, w_init[key].dtype, w[key].dtype)
                w_init[key] = w_init[key] + w[key]
        self.global_model.load_state_dict(w_init)