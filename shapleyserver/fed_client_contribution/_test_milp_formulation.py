import numpy as np
from scipy.optimize import milp
from scipy import optimize


# create a class for the MILP problem
class MILP_Shapley_prev():
    def __init__(self, selection_matrix, min_shapley_computation, max_shapley_computation=None, verbose=False):
        self.num_epochs = selection_matrix.shape[0]
        self.num_clients = selection_matrix.shape[1]
        self.selection_matrix = selection_matrix
        self.min_shapley_computation = min_shapley_computation
        if max_shapley_computation is None:
            self.max_shapley_computation = self.num_epochs
        else:
            self.max_shapley_computation = max_shapley_computation
        self.verbose = verbose

        # pre-compute
        self.build_objective()
        self.build_constraints()
        

    def build_objective(self):    
        # objective function: w followed by b, which we want to minimize
        self.objective = np.concatenate([np.ones(self.num_epochs), np.zeros(self.num_epochs*self.num_clients)])
    
    def build_constraints(self):

        # constraints 1: \sum_t b^t_i >= k for all i
        constraints_builder_1 = np.zeros((self.num_clients, self.num_epochs + self.num_epochs*self.num_clients))
        for client_index in range(self.num_clients):
            constraints_builder_1[client_index] = np.concatenate([np.zeros(self.num_epochs)] + 
                            [np.zeros(self.num_epochs) if i != client_index else self.selection_matrix[:, i] for i in range(self.num_clients)])
            


        # constraints 2: w^t * |i^t| - \sum_{i \in i^t} b^t_i >= 0 for all t
        constraints_builder_2 = np.zeros((self.num_epochs, self.num_epochs + self.num_epochs*self.num_clients))
        for time_index in range(self.num_epochs):
            time_vector = np.zeros(self.num_epochs)
            time_vector[time_index] = 1 * len(np.where(self.selection_matrix[time_index] == 1)[0])
            client_vector = np.zeros(self.num_epochs*self.num_clients)
            for client_index in range(self.num_clients):
                if self.selection_matrix[time_index][client_index] == 1:
                    client_vector[self.num_epochs*client_index + time_index] = -1
            constraints_builder_2[time_index] = np.concatenate([time_vector, client_vector])
      

        # constraints \sum_t w_t >= k
        # constraints_builder_3 = np.zeros((1, self.num_epochs + self.num_epochs*self.num_clients))
        # constraints_builder_3[0] = np.concatenate([np.ones(self.num_epochs), np.zeros(self.num_epochs*self.num_clients)])


        # self.constraint_builder = np.concatenate([constraints_builder_1, constraints_builder_2, constraints_builder_3])
        self.constraint_builder = np.concatenate([constraints_builder_1, constraints_builder_2])
        

    
    def build_upper_and_lower_bound(self):
        self.lower_bound = []
        self.upper_bound = []

        # constraints 1: \sum_t b^t_i >= k for all i
        for client_index in range(self.num_clients):
            self.lower_bound.append(self.min_shapley_computation)
            self.upper_bound.append(self.max_shapley_computation)

        # constraints 2: w^t * |i^t| - \sum_{i \in i^t} b^t_i >= 0 for all t
        for time_index in range(self.num_epochs):
            self.lower_bound.append(0)
            self.upper_bound.append(0)

        # constraints \sum_t w_t >= k
        # self.lower_bound.append(self.min_shapley_computation)
        # self.upper_bound.append(self.num_epochs)


        self.upper_bound = np.array(self.upper_bound)
        self.lower_bound = np.array(self.lower_bound)

        


    def solve(self):
        self.build_upper_and_lower_bound()
        bounds = optimize.Bounds(0, 1)
        integrality = np.concatenate([np.ones(self.num_epochs), np.zeros(self.num_epochs*self.num_clients)])
        constraints = optimize.LinearConstraint(A=self.constraint_builder, lb=self.lower_bound, ub=self.upper_bound)
        res = milp(c=self.objective, constraints=constraints,
           integrality=integrality, bounds=bounds)
        
        
        # print()
        # print("Solution")
        # print(res.success)
        if(res.success):
            if(self.verbose):
                print("---------Solution")
                print(f"Min epoch: {self.min_shapley_computation}")
                print(f"Max epoch: {self.max_shapley_computation}")
                print(f"optimal value: {res.fun}")
                print(f"optimal var: {res.x}")
                # print(f"constr: {self.constraint_builder @ res.x}")
                print(f"message: {res.message}")
            
            return res.success, res.fun, res.x[:self.num_epochs]
        else:
            return res.success, None, None
        
# a binary search algorithm to find the minimum number of epochs needed to compute the shapley values
def binary_search(selection_matrix, max_value=None, verbose=False):
    """
        Input:
            - selection_matrix: a matrix of shape (num_epochs, num_clients) where each row is the list of clients selected at each epoch

        Return: 
            - the minimum number of epochs needed to compute the shapley values
            - list of epochs where the shapley values are computed
            - minimum minimum number of time a client is selected and Shapley value is computed
    """

    # check never selected clients
    never_selected_clients = np.where(selection_matrix.sum(axis=0) == 0)[0]
    if verbose:
        print(f"Never selected clients: {never_selected_clients}")
    
    # delete never selected clients
    selection_matrix = np.delete(selection_matrix, never_selected_clients, axis=1)


    min_value = 1
    if max_value is None:
        max_value = selection_matrix.shape[0]
    mid = (min_value + max_value) // 2
    milp_shapley = MILP_Shapley_prev(selection_matrix, min_value, max_value, verbose=verbose)
    best_fun = None
    best_x = None
    # valid_x = []
    steps = 0
    while min_value < max_value:
        mid = (min_value + max_value) // 2
        if verbose:
            print(f"Min value: {min_value}, Max value: {max_value}, Mid value: {mid}")
        milp_shapley.min_shapley_computation = mid
        success, fun, x = milp_shapley.solve()
        if success:
            min_value = mid + 1
            best_fun = fun
            best_x = x
            # valid_x.append(x)
        else:
            max_value = mid
        if verbose and success:
            print()
        steps += 1

    if verbose:
        print(f"Steps: {steps}")
    # return valid_x
    return best_x



class MILP_Shapley_client_pos_neg():
    def __init__(self, selection_matrix, max_shapley_computation=None, gamma=0.5, weight_epochs=None, verbose=False):
        self.num_epochs = selection_matrix.shape[0]
        self.num_clients = selection_matrix.shape[1]
        self.selection_matrix = selection_matrix
        # self.min_shapley_computation = min_shapley_computation
        if max_shapley_computation is None:
            self.max_shapley_computation = self.num_epochs
        else:
            self.max_shapley_computation = max_shapley_computation
        self.gamma = gamma
        assert self.gamma >= 0 and self.gamma <= 1
        if(weight_epochs is None):
            self.weight_epochs = np.ones(self.num_epochs)
        else:
            self.weight_epochs = weight_epochs
        self.verbose = verbose

        
        

    def build_objective(self):    
        # objective function: w followed by b, which we want to minimize
        # self.objective = np.concatenate([np.ones(self.num_epochs), np.zeros(self.num_epochs*self.num_clients)])

        # maximize weights
        objective_epoch = (-1.0 / self.weight_epochs.shape[0]) * self.weight_epochs
        

        # maximize when a client is selected and Shapley is computed and
        # minimize when a client is not selected and Shapley is computed
        # To do that, for each variable, we need to add a negation variable
        objective_client = np.zeros(self.num_epochs * 2 * self.num_clients)
        
        for client_index in range(self.num_clients):
            num_selected_epochs = len(np.where(self.selection_matrix[:, client_index] == 1)[0])
            # num_non_selected_epochs = self.num_epochs - num_selected_epochs
            for epoch_index in range(self.num_epochs):
                # if(self.selection_matrix[epoch_index][client_index] == 1):
                #     objective_client[client_index * 2 * self.num_epochs + epoch_index * 2]  =  -1 / self.num_epochs
                #     objective_client[client_index * 2 * self.num_epochs + epoch_index * 2 + 1]  =  0
                # else:
                #     objective_client[client_index * 2 * self.num_epochs + epoch_index * 2]  =  0
                #     objective_client[client_index * 2 * self.num_epochs + epoch_index * 2 + 1]  =  1 / self.num_epochs
                # objective_client[client_index * 2 * self.num_epochs + epoch_index * 2]  =  -1 / self.num_epochs
                objective_client[client_index * 2 * self.num_epochs + epoch_index * 2]  =  -1 / num_selected_epochs
                # objective_client[client_index * 2 * self.num_epochs + epoch_index * 2 + 1]  =  1 / self.num_epochs
                pass
        # normalize
        objective_client = objective_client / self.num_clients

        self.objective = np.concatenate([(self.gamma) * objective_epoch, (1 - self.gamma) * objective_client])
        
    
    def build_constraints(self):
        # constraints \sum_t w_t <= k_max
        constraints_builder_1 = np.zeros((1, self.num_epochs + self.num_epochs * 2 * self.num_clients))
        constraints_builder_1[0] = np.concatenate([np.ones(self.num_epochs), np.zeros(self.num_epochs * 2 * self.num_clients)])
            


        # constraints 2: w^t * |i^t| - \sum_{i \in i^t} b^t_i >= 0 for all t
        constraints_builder_2 = np.zeros((self.num_epochs, self.num_epochs + self.num_epochs * 2 * self.num_clients))
        for time_index in range(self.num_epochs):
            time_vector = np.zeros(self.num_epochs)
            # time_vector[time_index] = 1 * len(np.where(self.selection_matrix[time_index] == 1)[0])
            time_vector[time_index] = 1 * self.num_clients
            client_vector = np.zeros(self.num_epochs * 2 * self.num_clients)
            for client_index in range(self.num_clients):
                if(self.selection_matrix[time_index][client_index] == 1):
                    client_vector[client_index * 2 * self.num_epochs + time_index * 2] = -1
                    client_vector[client_index * 2 * self.num_epochs + time_index * 2 + 1] = 0
                else:
                    client_vector[client_index * 2 * self.num_epochs + time_index * 2] = 0
                    client_vector[client_index * 2 * self.num_epochs + time_index * 2 + 1] = -1
                # if self.selection_matrix[time_index][client_index] == 1:
                #     client_vector[self.num_epochs*client_index + time_index] = -1
            constraints_builder_2[time_index] = np.concatenate([time_vector, client_vector])

        
        # # constraint 3: complementary variables
        constraints_builder_3 = np.zeros((self.num_epochs * self.num_clients, self.num_epochs + self.num_epochs * 2 * self.num_clients))
        for client_index in range(self.num_clients):
            for epoch_index in range(self.num_epochs):
                constraints_builder_3[client_index * self.num_epochs + epoch_index][self.num_epochs + client_index * 2 * self.num_epochs + epoch_index * 2] = 1
                constraints_builder_3[client_index * self.num_epochs + epoch_index][self.num_epochs + client_index * 2 * self.num_epochs + epoch_index * 2 + 1] = 1

      

        # self.constraint_builder = np.concatenate([constraints_builder_1, constraints_builder_2])
        self.constraint_builder = np.concatenate([constraints_builder_1, constraints_builder_2, constraints_builder_3])

        
        

    
    def build_upper_and_lower_bound(self):
        self.lower_bound = []
        self.upper_bound = []

        # constraints \sum_t w_t <= k_max
        self.lower_bound.append(1)
        self.upper_bound.append(self.max_shapley_computation)


        # constraints 2: w^t * |i^t| - \sum_{i \in i^t} b^t_i >= 0 for all t
        for time_index in range(self.num_epochs):
            self.lower_bound.append(0)
            self.upper_bound.append(0)


        # # constraint 3: complementary variables
        for client_index in range(self.num_clients):
            for epoch_index in range(self.num_epochs):
                self.lower_bound.append(1)
                self.upper_bound.append(1)

        

        self.upper_bound = np.array(self.upper_bound)
        self.lower_bound = np.array(self.lower_bound)

        


    def solve(self):

        self.build_upper_and_lower_bound()
        self.build_objective()
        self.build_constraints()

        
        # print(self.objective)
        # print()
        # print(self.constraint_builder)
        # print()
        

        bounds = optimize.Bounds(0, 1)
        integrality = np.concatenate([np.ones(self.num_epochs), np.zeros(self.num_epochs * 2 * self.num_clients)])
        constraints = optimize.LinearConstraint(A=self.constraint_builder, lb=self.lower_bound, ub=self.upper_bound)
        res = milp(c=self.objective, constraints=constraints,
           integrality=integrality, bounds=bounds)
        
        
        # print()
        # print("Solution")
        # print(res.success)
        if(res.success):
            if(self.verbose):
                print("---------Solution")
                # print(f"Min epoch: {self.min_shapley_computation}")
                print(f"Max epoch: {self.max_shapley_computation}")
                print(f"optimal value: {res.fun}")
                print(f"optimal valua weight: {self.objective[:self.num_epochs] @ res.x[:self.num_epochs]}")
                print(f"optimal value client: {self.objective[self.num_epochs:] @ res.x[self.num_epochs:]}")
                for client_index in range(self.num_clients):
                    print(f"client {client_index}: {res.x[self.num_epochs + client_index * 2 * self.num_epochs : self.num_epochs + (client_index + 1) * 2 * self.num_epochs]}")
                    print(f"optimal value for client {client_index}: {self.objective[self.num_epochs + client_index * 2 * self.num_epochs : self.num_epochs + (client_index + 1) * 2 * self.num_epochs] @ res.x[self.num_epochs + client_index * 2 * self.num_epochs : self.num_epochs + (client_index + 1) * 2 * self.num_epochs]}")
                print(f"optimal var: {res.x}")
                # print(f"constr: {self.constraint_builder @ res.x}")
                print(f"message: {res.message}")
            
            return res.success, res.fun, res.x[:self.num_epochs]
        else:
            return res.success, None, None
        


class MILP_Shapley():
    def __init__(self, selection_matrix, max_shapley_computation=None, gamma=0.5, weight_epochs=None, verbose=False):
        self.num_epochs = selection_matrix.shape[0]
        self.num_clients = selection_matrix.shape[1]
        self.selection_matrix = selection_matrix
        # self.min_shapley_computation = min_shapley_computation
        if max_shapley_computation is None:
            self.max_shapley_computation = self.num_epochs
        else:
            self.max_shapley_computation = max_shapley_computation
        self.gamma = gamma
        assert self.gamma >= 0 and self.gamma <= 1
        if(weight_epochs is None):
            self.weight_epochs = np.ones(self.num_epochs) / self.num_epochs
        else:
            self.weight_epochs = weight_epochs
        self.verbose = verbose

        
        

    def build_objective(self):    
        # objective function: w followed by b, which we want to minimize
        # self.objective = np.concatenate([np.ones(self.num_epochs), np.zeros(self.num_epochs*self.num_clients)])

        # maximize weights
        # assert self.weight_epochs.sum() == 1
        objective_epoch = self.weight_epochs * -1

        # print("sum", objective_epoch.sum())
        # print(self.weight_epochs)
        # print(self.weight_epochs.shape[0])
        # print(objective_epoch)
        

        # maximize when a client is selected and Shapley is computed and
        # minimize when a client is not selected and Shapley is computed
        # To do that, for each variable, we need to add a negation variable
        objective_client = np.zeros(self.num_epochs * 2 * self.num_clients)
        
        for client_index in range(self.num_clients):
            num_selected_epochs = len(np.where(self.selection_matrix[:, client_index] == 1)[0])
            # num_non_selected_epochs = self.num_epochs - num_selected_epochs
            for epoch_index in range(self.num_epochs):
                if(self.selection_matrix[epoch_index][client_index] == 1):
                    objective_client[client_index * 2 * self.num_epochs + epoch_index * 2]  =  -1 / num_selected_epochs
                #     objective_client[client_index * 2 * self.num_epochs + epoch_index * 2 + 1]  =  0
                # else:
                #     objective_client[client_index * 2 * self.num_epochs + epoch_index * 2]  =  0
                #     objective_client[client_index * 2 * self.num_epochs + epoch_index * 2 + 1]  =  1 / self.num_epochs
                # objective_client[client_index * 2 * self.num_epochs + epoch_index * 2]  =  -1 / self.num_epochs
                # objective_client[client_index * 2 * self.num_epochs + epoch_index * 2]  =  -1 / num_selected_epochs
                # objective_client[client_index * 2 * self.num_epochs + epoch_index * 2 + 1]  =  1 / self.num_epochs
                pass
        # normalize
        objective_client = objective_client / self.num_clients

        self.objective = np.concatenate([(self.gamma) * objective_epoch, (1 - self.gamma) * objective_client])
        # print(self.objective[:self.num_epochs].sum())
        # print(self.objective[self.num_epochs:].sum())
        # for client_index in range(self.num_clients):
        #     print(self.objective[self.num_epochs + client_index * 2 * self.num_epochs : self.num_epochs + (client_index + 1) * 2 * self.num_epochs].sum())
    
    def build_constraints(self):
        # constraints \sum_t w_t <= k_max
        constraints_builder_1 = np.zeros((1, self.num_epochs + self.num_epochs * 2 * self.num_clients))
        constraints_builder_1[0] = np.concatenate([np.ones(self.num_epochs), np.zeros(self.num_epochs * 2 * self.num_clients)])
            


        # constraints 2: w^t * |i^t| - \sum_{i \in i^t} b^t_i >= 0 for all t
        constraints_builder_2 = np.zeros((self.num_epochs, self.num_epochs + self.num_epochs * 2 * self.num_clients))
        for time_index in range(self.num_epochs):
            time_vector = np.zeros(self.num_epochs)
            time_vector[time_index] = 1 * len(np.where(self.selection_matrix[time_index] == 1)[0])
            # time_vector[time_index] = 1 * self.num_clients
            client_vector = np.zeros(self.num_epochs * 2 * self.num_clients)
            for client_index in range(self.num_clients):
                if(self.selection_matrix[time_index][client_index] == 1):
                    client_vector[client_index * 2 * self.num_epochs + time_index * 2] = -1
                    client_vector[client_index * 2 * self.num_epochs + time_index * 2 + 1] = 0
                # else:
                #     client_vector[client_index * 2 * self.num_epochs + time_index * 2] = 0
                #     client_vector[client_index * 2 * self.num_epochs + time_index * 2 + 1] = -1
                # if self.selection_matrix[time_index][client_index] == 1:
                #     client_vector[self.num_epochs*client_index + time_index] = -1
            constraints_builder_2[time_index] = np.concatenate([time_vector, client_vector])

        
        # # constraint 3: complementary variables
        # constraints_builder_3 = np.zeros((self.num_epochs * self.num_clients, self.num_epochs + self.num_epochs * 2 * self.num_clients))
        # for client_index in range(self.num_clients):
        #     for epoch_index in range(self.num_epochs):
        #         constraints_builder_3[client_index * self.num_epochs + epoch_index][self.num_epochs + client_index * 2 * self.num_epochs + epoch_index * 2] = 1
        #         constraints_builder_3[client_index * self.num_epochs + epoch_index][self.num_epochs + client_index * 2 * self.num_epochs + epoch_index * 2 + 1] = 1

      

        self.constraint_builder = np.concatenate([constraints_builder_1, constraints_builder_2])
        # self.constraint_builder = np.concatenate([constraints_builder_1, constraints_builder_2, constraints_builder_3])

        
        

    
    def build_upper_and_lower_bound(self):
        self.lower_bound = []
        self.upper_bound = []

        # constraints \sum_t w_t <= k_max
        self.lower_bound.append(1)
        self.upper_bound.append(self.max_shapley_computation)


        # constraints 2: w^t * |i^t| - \sum_{i \in i^t} b^t_i >= 0 for all t
        for time_index in range(self.num_epochs):
            self.lower_bound.append(0)
            self.upper_bound.append(0)


        # # constraint 3: complementary variables
        # for client_index in range(self.num_clients):
        #     for epoch_index in range(self.num_epochs):
        #         self.lower_bound.append(1)
        #         self.upper_bound.append(1)

        

        self.upper_bound = np.array(self.upper_bound)
        self.lower_bound = np.array(self.lower_bound)

        


    def solve(self):

        
        self.build_upper_and_lower_bound()
        self.build_objective()
        self.build_constraints()

        # print(self.selection_matrix)
        # print(self.objective)
        # print()
        # print(self.constraint_builder)
        # print()
        

        bounds = optimize.Bounds(0, 1)
        integrality = np.concatenate([np.ones(self.num_epochs), np.zeros(self.num_epochs * 2 * self.num_clients)])
        constraints = optimize.LinearConstraint(A=self.constraint_builder, lb=self.lower_bound, ub=self.upper_bound)
        res = milp(c=self.objective, constraints=constraints,
           integrality=integrality, bounds=bounds)
        
        
        # print()
        # print("Solution")
        # print(res.success)
        if(res.success):
            if(self.verbose):
                print("---------Solution")
                # print(f"Min epoch: {self.min_shapley_computation}")
                print(f"Max epoch: {self.max_shapley_computation}")
                print(f"optimal value: {res.fun}")
                print(f"optimal valua weight: {self.objective[:self.num_epochs] @ res.x[:self.num_epochs]}")
                print(f"optimal value client: {self.objective[self.num_epochs:] @ res.x[self.num_epochs:]}")
                for client_index in range(self.num_clients):
                    print(f"client {client_index}: {res.x[self.num_epochs + client_index * 2 * self.num_epochs : self.num_epochs + (client_index + 1) * 2 * self.num_epochs]}")
                    # print(f"weight: {self.objective[self.num_epochs + client_index * 2 * self.num_epochs : self.num_epochs + (client_index + 1) * 2 * self.num_epochs]}")
                    print(f"optimal value for client {client_index}: {self.objective[self.num_epochs + client_index * 2 * self.num_epochs : self.num_epochs + (client_index + 1) * 2 * self.num_epochs] @ res.x[self.num_epochs + client_index * 2 * self.num_epochs : self.num_epochs + (client_index + 1) * 2 * self.num_epochs]}")
                print(f"optimal var: {res.x}")
                # print(f"constr: {self.constraint_builder @ res.x}")
                print(f"message: {res.message}")
            
            return res.success, res.fun, res.x[:self.num_epochs]
        else:
            return res.success, None, None