import numpy as np
from functools import reduce
from tqdm import tqdm, trange
import random
import time
from itertools import chain, combinations
import operator as op
import logging





def call_shapley_computation_method(args, game, logger):
    args['approximation_method'] = 'comp_contrib'

    m = 50 * game.n
    shapley_value = shapley_comp_contrib(game, m)
    #logger.info(f"Comp contrib: {shapley_value}")
    #print(f"Comp contrib: {shapley_value}")
    logging.info("Comp contrib: {}".format(shapley_value))

    # if(args.approximation_method == 'monte_carlo'):
    #     m = 100
    #     shapley_value = shapley_monte_carlo(game, m)
    #     logger.info(f"Monte carlo: {shapley_value}")
        

    # elif(args.approximation_method == 'monte_carlo_hoeffding'):
    #     m = 1000
    #     shapley_value = shapley_monte_carlo_hoeffding(game, m)
    #     logger.info(f"Monte carlo hoeffding: {shapley_value}")
        

    """  elif(args.approximation_method == 'exact'):                
        shapley_value = shapley_exact(game)
        logger.info(f"Exact: {shapley_value}")
        
    elif(args.approximation_method == 'exact_own'):
        shapley_value = shapley_exact_own(game)
        logger.info(f"Exact own: {shapley_value}")
                
    elif(args.approximation_method == 'comp_contrib'):  #RUN THIS
        m = 50 * game.n
        shapley_value = shapley_comp_contrib(game, m)
        #logger.info(f"Comp contrib: {shapley_value}")
        print(f"Comp contrib: {shapley_value}")

    else:
        raise ValueError("Unknown Shapley value approximation method") """
    
    #print(f"Shapley value sum for each utility: {[sum(list(shapley_value[i].values())) for i in range(2)]}")
    logging.info("Shapley value sum for each utility: {}".format([sum(list(shapley_value[i].values())) for i in range(2)]))
    return shapley_value


# def isnotconverge_gtg(self, k):
#     # converge paras
#     CONVERGE_MIN_K = 3*10
#     last_k = 10
#     CONVERGE_CRITERIA = 0.05
    
#     if k <= CONVERGE_MIN_K:
#         return True
#     all_vals = (np.cumsum(self.Contribution_records, 0) /
#                 np.reshape(np.arange(1, len(self.Contribution_records)+1), (-1, 1)))[-self.last_k:]
#     errors = np.mean(np.abs(
#         all_vals[-self.last_k:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
#     if np.max(errors) > self.CONVERGE_CRITERIA:
#         return True
#     return False

# def shaply_value_gtg_paper(game):
#         idxs = game.selected_clients
#         N_all = game._n_all
#         N = len(idxs)
#         Contribution_records = []

#         # sets = list(powerset(idxs))

#         util = {}
#         S_0 = ()
#         util[S_0] = game.eval_utility(S_0)[self.utility_index]

#         # S_all = sets[-1]
#         S_all = tuple(idxs)
#         util[S_all] = game.eval_utility(S_all)[self.utility_index]

        

#         k = 0
#         while isnotconverge(k):
#             for pi in idxs:
#                 k += 1
#                 v = [0 for i in range(N+1)]
#                 v[0] = util[S_0]
#                 marginal_contribution_k = {idx: 0 for idx in range(N_all)}

#                 idxs_k = np.concatenate(
#                     (np.array([pi]), np.random.permutation([p for p in idxs if p != pi])))

#                 for j in range(1, N+1):
#                     # key = C subset
#                     C = idxs_k[:j]
#                     C = tuple(np.sort(C, kind='mergesort'))

#                     # truncation
#                     if abs(util[S_all] - v[j-1]) >= self.eps:
#                         if util.get(C) != None:
#                             v[j] = util[C]
#                         else:
#                             v[j] = game.eval_utility(C)[self.utility_index]
#                     else:
#                         v[j] = v[j-1]

#                     # record calculated V(C)
#                     util[C] = v[j]

#                     # update SV
#                     marginal_contribution_k[idxs_k[j-1]] = v[j] - v[j-1]

#                 self.Contribution_records.append(
#                     [marginal_contribution_k[i] for i in range(N_all)])

#         # shapley value calculation
#         shapley_value = (np.cumsum(self.Contribution_records, 0) /
#                          np.reshape(np.arange(1, len(self.Contribution_records)+1), (-1, 1)))[-1:].tolist()[0]

#         # store round t results
#         self.SV_t[t] = {key: sv for key, sv in enumerate(shapley_value)}

#         self.Ut[t] = copy.deepcopy(util)
#         # print("Utility index:", self.utility_index)
#         # print(self.SV_t[t])
#         # print(sum(self.SV_t[t].values()))
#         return self.SV_t[t]

        




# function for generating the power set
def powerset(iterable):
    s = list(iterable)
    l = chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))
    return {tuple(sorted(tmp)): i for i, tmp in enumerate(l)}


# function for computing the combinatorial number
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom



def shapley_exact_own(game):
    n = game.n
    shapley_value = game.default_shapley_value
    for client_id in game.selected_clients:
        sublist = list(game.selected_clients).copy()
        sublist.pop(sublist.index(client_id))
        subpowerset = powerset(sublist)
        for s in tqdm(subpowerset.keys()):
            v1 = game.eval_utility(s)
            v2 = game.eval_utility(list(s)+[client_id])
            # if(client_id == 4):
            # if(v2[1] == v1[1]):
            #     print(list(s), list(s)+[client_id])
            #     print(v1[1], v2[1], v2[1] - v1[1])
            # print()
            for i in range(game.utility_dim):
                val = (v2[i] - v1[i]) / ncr(n - 1, len(s))
                shapley_value[i][client_id] += val
        # (v({a}) - v({})) / (N-1)
        v = game.eval_utility([client_id])
        # if(client_id == 4):
        # print(f"null for client {client_id} utility: {v}, normalized: {[v[0]/n, v[1]/n]}")
        for i in range(game.utility_dim):
            shapley_value[i][client_id] += v[i]
            shapley_value[i][client_id] /= n
    # print(f"Naive: {shapley_value}")
    return shapley_value


def shapley_exact(game):
    all_participants = game.selected_clients
    n = game.n
    shapley_value = game.default_shapley_value
    coef = {client_id: 0 for client_id in all_participants} # coeff is same for all utility_dim
    fact = np.math.factorial
    for s in trange(n):
        coef[s] = fact(s) * fact(n - s - 1) / fact(n)
    sets = list(powerset(all_participants))
    for idx in trange(len(sets)):
        u = game.eval_utility(sets[idx])
        for i in range(game.utility_dim):
            for j in sets[idx]:
                shapley_value[i][j] += coef[len(sets[idx]) - 1] * u[i]
            for j in set(all_participants) - set(sets[idx]):
                shapley_value[i][j] -= coef[len(sets[idx])] * u[i]

    
    return shapley_value

def get_selection_dict(num_clients, idxs_participating_clients):
    selection_dict = {}
    for i in range(num_clients):
        selection_dict[i] = False
    for i in idxs_participating_clients:
        selection_dict[i] = True
    return selection_dict


def split_permutation(m, num) -> np.ndarray:
    """Split permutation
    """
    assert m > 0
    quotient = int(m / num)
    remainder = m % num

    perm_arr = []
    r = []
    for i in range(m):
        r.append(i)
        if (remainder > 0
                and len(r) == quotient + 1) or (remainder <= 0
                                                and len(r) == quotient):
            remainder -= 1
            perm_arr.append(r)
            r = []
    return perm_arr


def split_permutation_num(m, num) -> np.ndarray:
    """
        Split permutation num
    """
    assert m > 0
    quotient = int(m / num)
    remainder = m % num
    if remainder > 0:
        perm_arr = [quotient] * (num - remainder) + [quotient + 1] * remainder
    else:
        perm_arr = [quotient] * num
    return np.asarray(perm_arr)


def shapley_monte_carlo(game, m):
    """
        Compute Shapley value by sampling local_m permutations
    """
    n = game.n
    local_state = np.random.RandomState(None)
    shapley_value = game.default_shapley_value
    idxs = game.selected_clients

    for _ in trange(m):
        local_state.shuffle(idxs) 
        old_u = [0, 0] # utility of empty set
        for j in range(1, n + 1):
            temp_u = game.eval_utility(idxs[:j])
            for i in range(game.utility_dim):
                contribution = temp_u[i] - old_u[i]
                shapley_value[i][idxs[j - 1]] += contribution
                old_u[i] = temp_u[i]
    for i in range(game.utility_dim):
        for j in idxs:
            shapley_value[i][j] /= m    
    return shapley_value



def _cc_shap_task(game, local_m):
    """
        Compute Shapley value by sampling local_m complementary contributions
    """
    n = game.n
    local_state = np.random.RandomState(None)
    utility = [np.zeros((n + 1, n)) for _ in range(game.utility_dim)]
    count = np.zeros((n + 1, n))
    idxs = np.arange(n)
    selected_clients = np.array(game.selected_clients)
    
    for _ in trange(local_m):
        local_state.shuffle(idxs)
        j = random.randint(1, n) # random split with at least one client in each group
        u_1 = game.eval_utility(selected_clients[idxs[:j]])
        u_2 = game.eval_utility(selected_clients[idxs[j:]])

        
        temp = np.zeros(n)
        temp[idxs[:j]] = 1
        count[j, :] += temp
        for i in range(game.utility_dim):
            utility[i][j, :] += temp * (u_1[i] - u_2[i])
        
        temp = np.zeros(n)
        temp[idxs[j:]] = 1
        count[n - j, :] += temp
        for i in range(game.utility_dim):
            utility[i][n - j, :] += temp * (u_2[i] - u_1[i])
            

    return utility, count

def split_num(m_list, num) -> np.ndarray:
    """Split num
    """
    perm_arr_list = None
    local_state = np.random.RandomState(None)
    for m in m_list:
        assert m >= 0
        if m != 0:
            m = int(m)
            quotient = int(m / num)
            remainder = m % num
            if remainder > 0:
                perm_arr = [[quotient]] * (num - remainder) + [[quotient + 1]
                                                               ] * remainder
                local_state.shuffle(perm_arr)
            else:
                perm_arr = [[quotient]] * num
        else:
            perm_arr = [[0]] * num

        if perm_arr_list is None:
            perm_arr_list = perm_arr
        else:
            perm_arr_list = np.concatenate((perm_arr_list, perm_arr), axis=-1)

    return np.asarray(perm_arr_list)

def shapley_comp_contrib(game, m, proc_num=1):
    """
        Compute Shapley value by sampling m complementary contributions
    """
    if proc_num < 0:
        raise ValueError('Invalid proc num.')

    n = game.n
    utility, count = _cc_shap_task(game, m)
    shapley_value = [np.zeros(n) for _ in range(game.utility_dim)]
    # shapley_value = game.default_shapley_value
    
    for i in range(n + 1):
        for j in range(n):
            for k in range(game.utility_dim):
                shapley_value[k][j] += 0 if count[i][j] == 0 else (utility[k][i][j] / count[i][j])
    
    for i in range(game.utility_dim):
        shapley_value[i] /= n
        shapley_value[i] = {game.selected_clients[idx]: value for idx, value in enumerate(shapley_value[i])}

    # get default Shapley value for non-selected clients
    precomputed_shapley_value = game.default_shapley_value
    for i in range(game.utility_dim):
        for client_id in precomputed_shapley_value[i]:
            if(game.client_selection_vector[client_id]):
                assert client_id in shapley_value[i]
                precomputed_shapley_value[i][client_id] = shapley_value[i][client_id]

    return precomputed_shapley_value



# def _mch_task(game, mk):
#     """Compute Shapley value by sampling mk(list) marginal contributions
#     """

#     n = game.n
#     utility = [np.zeros((n, n)) for _ in range(game.utility_dim)]
#     local_state = np.random.RandomState(None)
#     for i in trange(n):
#         idxs = []
#         for _ in range(n):
#             if _ is not i:
#                 idxs.append(_)
#         for j in range(n):
#             for _ in range(int(mk[j])):
#                 local_state.shuffle(idxs)
#                 # u_1 = utility_map[frozenset(idxs[:j])]
#                 # u_2 = utility_map[frozenset(idxs[:j] + [i])]
#                 u_1 = game.eval_utility(idxs[:j])
#                 u_2 = game.eval_utility(idxs[:j] + [i])
#                 for k in range(game.utility_dim):
#                     utility[k][i][j] += u_2[k] - u_1[k]

#     return utility






