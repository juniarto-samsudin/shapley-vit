import numpy as np
from scipy.optimize import milp
from scipy import optimize
from scipy.spatial.distance import pdist

        
# constraint problem 1: focus on accuracy coverage and epochs with higher participating clients
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

        # normalized selection matrix
        normalized_selection_matrix = self.selection_matrix / self.selection_matrix.sum(axis=0)

        client_weight = normalized_selection_matrix.sum(axis=1)
        client_weight = client_weight / client_weight.sum()
        # print(client_weight)

        # update weight epochs
        self.weight_epochs = self.weight_epochs * self.gamma + client_weight * (1 - self.gamma)
        self.verbose = verbose
        if(self.verbose):
            print(f"weight epochs: {self.weight_epochs}")

           

    def build_objective(self):    
        objective_epoch = self.weight_epochs * -1
        self.objective = objective_epoch

          
    def build_constraints(self):
        # constraints \sum_t w_t <= k_max
        constraints_builder_1 = np.ones((1, self.num_epochs))
        self.constraint_builder = np.concatenate([constraints_builder_1])
        

    
    def build_upper_and_lower_bound(self):
        self.lower_bound = []
        self.upper_bound = []

        # constraints \sum_t w_t <= k_max
        self.lower_bound.append(1)
        self.upper_bound.append(self.max_shapley_computation)        

        self.upper_bound = np.array(self.upper_bound)
        self.lower_bound = np.array(self.lower_bound)

        


    def solve(self):

        
        self.build_upper_and_lower_bound()
        self.build_objective()
        self.build_constraints()

        bounds = optimize.Bounds(0, 1)
        integrality = np.concatenate([np.ones(self.num_epochs)])
        constraints = optimize.LinearConstraint(A=self.constraint_builder, lb=self.lower_bound, ub=self.upper_bound)
        res = milp(c=self.objective, constraints=constraints,
           integrality=integrality, bounds=bounds)
        
        
        if(res.success):
            if(self.verbose):
                print("---------Solution")
                print(f"Max epoch: {self.max_shapley_computation}")
                print(f"optimal value: {res.fun}")
                print(f"optimal valua weight: {self.objective[:self.num_epochs] @ res.x[:self.num_epochs]}")
                print(f"optimal var: {res.x}")
                print(f"message: {res.message}")
            
            return res.success, res.fun, res.x[:self.num_epochs]
        else:
            return res.success, None, None



# constraint problem 3: minimize absolute difference between the selection matrix and the uniform distribution
class MILP_Shapley_Two_Sided():
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

        self.auxialiary_variable_dim = int(self.num_clients * (self.num_clients - 1) / 2)
        self.weight_epochs = self.weight_epochs
        # print(self.weight_epochs)
        self.verbose = verbose

           

    def build_objective(self):    
        objective_epoch = self.weight_epochs * -1
        objective_pairwise_distance = np.ones(self.auxialiary_variable_dim) / self.auxialiary_variable_dim
        self.objective = np.concatenate([(self.gamma) * objective_epoch, (1 - self.gamma) * objective_pairwise_distance])
        if(self.verbose):
            print(f"Objective: {self.objective}")
          
    def build_constraints(self):
        # constraints \sum_t w_t <= k_max
        constraints_builder_1 = np.zeros((1, self.num_epochs + self.auxialiary_variable_dim))
        constraints_builder_1[0] = np.concatenate([np.ones(self.num_epochs), np.zeros(self.auxialiary_variable_dim)])
        # print(constraints_builder_1)       
       
       
        # normalized selection matrix
        constraints_builder_2 = []
        normalized_selection_matrix = self.selection_matrix / self.selection_matrix.sum(axis=0)
        auxialiary_variable_index = 0
        # print(normalized_selection_matrix)
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                diff_vector = []
                for t in range(self.num_epochs):
                    diff_vector.append((normalized_selection_matrix[t][i] - normalized_selection_matrix[t][j]) / self.num_clients)

                auxialiary_variable_builder = [0 for _ in range(self.auxialiary_variable_dim)]
                auxialiary_variable_builder[auxialiary_variable_index] = 1
                constraints_builder_2.append([-1 * x for x in diff_vector] + auxialiary_variable_builder)
                constraints_builder_2.append(diff_vector + auxialiary_variable_builder)
                auxialiary_variable_index += 1

        
        if(self.verbose):
            print(f"selection matrix:\n{self.selection_matrix}")
            print(f"Normalized selection matrix:\n{normalized_selection_matrix}")
            
        self.constraint_builder = np.concatenate([constraints_builder_1, constraints_builder_2])
        
        if(self.verbose):
            print(f"Constraints builder:\n{self.constraint_builder}")

    
    def build_upper_and_lower_bound(self):
        self.lower_bound = []
        self.upper_bound = []

        # constraints \sum_t w_t <= k_max
        self.lower_bound.append(1)
        self.upper_bound.append(self.max_shapley_computation)

        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                self.lower_bound.append(0)
                self.upper_bound.append(1)
                self.lower_bound.append(0)
                self.upper_bound.append(1)     

        self.upper_bound = np.array(self.upper_bound)
        self.lower_bound = np.array(self.lower_bound)



    def solve(self):

        
        self.build_upper_and_lower_bound()
        self.build_objective()
        self.build_constraints()

        bounds = optimize.Bounds(0, 1)
        integrality = np.concatenate([np.ones(self.num_epochs), np.zeros(self.auxialiary_variable_dim)])
        constraints = optimize.LinearConstraint(A=self.constraint_builder, lb=self.lower_bound, ub=self.upper_bound)
        res = milp(c=self.objective, constraints=constraints,
           integrality=integrality, bounds=bounds)
        
        
        if(res.success):
            if(self.verbose):
                print("---------Solution")
                print(f"Max epoch: {self.max_shapley_computation}")
                print(f"optimal value: {res.fun}")
                print(f"optimal valua weight: {self.objective[:self.num_epochs] @ res.x[:self.num_epochs]}")
                print(f"optimal var: {res.x}")
                print(f"message: {res.message}")
            
            return res.success, res.fun, res.x[:self.num_epochs]
        else:
            return res.success, None, None


# constraint problem 3: Maximizing the lower bound of constraint 2
class MILP_Shapley_Two_Sided_Approx():
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
        self.verbose = verbose
        assert self.gamma >= 0 and self.gamma <= 1
        if(weight_epochs is None):
            self.weight_epochs = np.ones(self.num_epochs) / self.num_epochs
        else:
            self.weight_epochs = weight_epochs

        # normalized selection matrix
        normalized_selection_matrix = self.selection_matrix / self.selection_matrix.sum(axis=0)
        absolute_diff = []
        sum_absolute_diff = 0
        for i in range(self.num_epochs):
            absolute_diff.append((pdist(normalized_selection_matrix[i].reshape(-1, normalized_selection_matrix[i].shape[0]).T)).sum())
            sum_absolute_diff += absolute_diff[-1]

        if(self.verbose):
            print(f"Normalized selection matrix:\n{normalized_selection_matrix}")
            print(f"Absolute difference: {absolute_diff}")
        # normalize absolute difference
        absolute_diff = np.array(absolute_diff) / sum_absolute_diff

        if(self.verbose):        
            print(f"Normalized absolute difference: {absolute_diff}")
            print(f"Weight epochs: {self.weight_epochs}")
        # update weight epochs. Penalize for the difference between the selection matrix and the uniform distribution
        self.weight_epochs = self.weight_epochs * self.gamma - absolute_diff * (1 - self.gamma)
        if(self.verbose):
            print(f"Updated weight epochs: {self.weight_epochs}")
        
        
           

    def build_objective(self):    
        objective_epoch = self.weight_epochs * -1
        self.objective = objective_epoch

          
    def build_constraints(self):
        # constraints \sum_t w_t <= k_max
        constraints_builder_1 = np.ones((1, self.num_epochs))
        self.constraint_builder = np.concatenate([constraints_builder_1])
        

    
    def build_upper_and_lower_bound(self):
        self.lower_bound = []
        self.upper_bound = []

        # constraints \sum_t w_t <= k_max
        self.lower_bound.append(1)
        self.upper_bound.append(self.max_shapley_computation)        

        self.upper_bound = np.array(self.upper_bound)
        self.lower_bound = np.array(self.lower_bound)

        


    def solve(self):

        
        self.build_upper_and_lower_bound()
        self.build_objective()
        self.build_constraints()

        bounds = optimize.Bounds(0, 1)
        integrality = np.concatenate([np.ones(self.num_epochs)])
        constraints = optimize.LinearConstraint(A=self.constraint_builder, lb=self.lower_bound, ub=self.upper_bound)
        res = milp(c=self.objective, constraints=constraints,
           integrality=integrality, bounds=bounds)
        
        
        if(res.success):
            if(self.verbose):
                print("---------Solution")
                print(f"Max epoch: {self.max_shapley_computation}")
                print(f"optimal value: {res.fun}")
                print(f"optimal valua weight: {self.objective[:self.num_epochs] @ res.x[:self.num_epochs]}")
                print(f"optimal var: {res.x}")
                print(f"message: {res.message}")
            
            return res.success, res.fun, res.x[:self.num_epochs]
        else:
            return res.success, None, None





# selection_matrix = np.array(
#     [
#         [0, 1, 1, 0, 0],
#         [1, 0, 1, 0, 0],
#         [0, 0, 0, 1, 1],
#         [1, 0, 0, 0, 1],
#     ]
# )


# # selection_matrix = np.array(
# #     [ 
# #         [1, 0, 1],
# #         [0, 1, 1],
# #         [1, 0, 1],
# #     ]
# # )


# # selection_matrix = np.array(
# #     [ 
# #         [1, 0],
# #         [0, 1],
# #         [0, 1],
# #         [1, 0],
# #     ]
# # )


# gamma = 0.5
# verbose = False
# max_shapley_computation = selection_matrix.shape[0] - 1
# milp_shapley = MILP_Shapley(selection_matrix=selection_matrix, 
#                                      max_shapley_computation=max_shapley_computation, 
#                                      gamma=gamma,
#                                      verbose=False)
# print(milp_shapley.solve())
# print()




# max_shapley_computation = selection_matrix.shape[0] - 1
# milp_shapley = _MILP_Shapley_Two_Sided(selection_matrix=selection_matrix, 
#                                      max_shapley_computation=max_shapley_computation, 
#                                      gamma=gamma,
#                                      verbose=False)
# print(milp_shapley.solve())
# print()

# max_shapley_computation = selection_matrix.shape[0] - 1
# milp_shapley = MILP_Shapley_Two_Sided(selection_matrix=selection_matrix, 
#                                      max_shapley_computation=max_shapley_computation, 
#                                      gamma=gamma,
#                                      verbose=False)
# print(milp_shapley.solve())

[ 
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0],
]