
import copy
from wolframclient.language import wlexpr
from wolframclient.evaluation import WolframLanguageSession, SecuredAuthenticationKey, WolframCloudSession
import numpy as np
import time
from tqdm import trange, tqdm
from scipy.special import comb
from fed_client_contribution.utils_shapley import powerset, ncr


"""
    ComFedSV
"""


def comfedsv(args, utility_matrix, all_subsets):
    T = args.rounds
    N = args.num_clients
    shapley_value_per_round = []
    computation_time_per_round = []
    for t in tqdm(range(T)):
        s_time = time.time()
        valuation_completed = {client_id: 0 for client_id in range(N)}
        for client_id in range(N):
            sublist = list(range(N))
            sublist.pop(sublist.index(client_id))
            subpowerset = powerset(sublist)
            for s in subpowerset.keys():
                v1 = utility_matrix[t][all_subsets[s]]
                v2 = utility_matrix[t][all_subsets[tuple(
                    sorted(list(s)+[client_id]))]]
                val = (v2 - v1) / ncr(N - 1, len(s))
                valuation_completed[client_id] += val

            # (v({a}) - v({})) / (N-1)
            valuation_completed[client_id] += utility_matrix[t][all_subsets[tuple([
                                                                                  client_id])]]

            # outer normalization
            valuation_completed[client_id] /= N
        shapley_value_per_round.append(valuation_completed)
        computation_time_per_round.append(time.time() - s_time)
    return shapley_value_per_round, computation_time_per_round


def call_comfedsv(game, all_subsets, logger):
    # print(all_subsets)
    # print(game.selected_clients)
    utilities = [np.zeros(len(all_subsets)) for i in range(game.utility_dim)]
    sets = list(powerset(game.selected_clients))
    for idx in trange(len(sets)):
        u = game.eval_utility(sets[idx])
        for i in range(game.utility_dim):
            utilities[i][all_subsets[sets[idx]]] = u[i]

    # print(utilities)
    # print(roundly_mask(game.selected_clients, all_subsets))
    # # quit()

    return utilities, roundly_mask(game.selected_clients, all_subsets)

# mask in each round


def roundly_mask(idxs_users, all_subsets):
    mask_vec = np.zeros(len(all_subsets))
    subpowerset = powerset(idxs_users)
    for s in subpowerset.keys():
        i = all_subsets[s]
        mask_vec[i] = 1
    return mask_vec


# from typing import Callable,Any
# import SV_algs.shapley_utils
# from SV_algs.shapley_utils import powersettool
# from scipy.special import comb


def shapley_value(utility, game):
    N = len(game.selected_clients)
    sv_dict = {id: 0 for id in range(game._n_all)}
    for S in utility.keys():
        if S != ():
            for id in S:
                marginal_contribution = utility[S] - \
                    utility[tuple(i for i in S if i != id)]
                sv_dict[id] += marginal_contribution / \
                    ((comb(N-1, len(S)-1))*N)
    return sv_dict



class ShapleyValue:
    def __init__(self):
        self.FL_name = 'Null'
        self.SV = {}  # dict: {id:SV,...}


"""
    Fed_SV
"""


class Fed_SV(ShapleyValue):
    def __init__(self, utility_index):
        super().__init__()
        self.Ut = {}
        self.SV_t = {}
        self.utility_index = utility_index

        # TMC paras
        self.Contribution_records = []

        # converge paras
        self.CONVERGE_MIN_K = 200
        self.last_k = 10
        self.CONVERGE_CRITERIA = 0.05

    def compute_shapley_value(self, game, t):
        idxs = list(range(game._n_all))
        N = len(idxs)
        sets = list(powerset(idxs))

        util = {}
        S_0 = ()
        util[S_0] = game.eval_utility(S_0)[self.utility_index]

        S_all = sets[-1]
        util[S_all] = game.eval_utility(S_all)[self.utility_index]

        # group test relate
        last_uds = []
        Z = 0
        for n in range(1, N):
            Z += 1/n
        Z *= 2
        UD = np.zeros([N, N], dtype=np.float32)
        p = np.array([N/(i*(N-i)*Z) for i in range(1, N)])

        k = 0
        while self.isnotconverge_Group(last_uds, UD) or k < self.CONVERGE_MIN_K:
            k += 1
            len_k = 0
            # 1. draw len_K ~ q(len_k)
            len_k = np.random.choice(np.arange(1, N), p=p)

            # 2. sample S with len_k
            S = np.random.choice(idxs, size=len_k, replace=False)

            # 3. M(S) + V(S)
            S = tuple(np.sort(S, kind='mergesort'))
            if util.get(S) != None:
                u_S = util[S]
            else:
                # u_S = V_S_t(t=t, S=S)
                u_S = game.eval_utility(S)[self.utility_index]

            # 4. Group Testing update UD
            UD = (k-1)/k*UD

            for i in range(0, N):
                for j in range(0, N):
                    delta_beta = S.count(i+1)-S.count(j+1)
                    if delta_beta != 0:
                        value = delta_beta*u_S*Z/k
                        UD[i, j] += value

            last_uds.append(UD)

        u_N = util[S_all]

        # timer
        st = time.time()
        # timer

        shapley_value = self.solveFeasible(N, u_N, UD)

        # timer
        dura = time.time()-st
        print('Solve Feasible using %.3f seconds' % dura)
        # timer

        self.Ut[t] = copy.deepcopy(util)
        self.SV_t[t] = {key+1: sv for key, sv in enumerate(shapley_value)}

        return self.SV_t[t]

    def isnotconverge_Group(self, last_uds, UD):
        if len(last_uds) <= self.CONVERGE_MIN_K:
            return True
        for i in range(-self.last_k, 0):
            ele = last_uds[i]
            delta = np.sum(np.abs(UD-ele), axis=(0, 1))/len(UD[0])
            if delta > self.CONVERGE_CRITERIA:
                return True
        return False

    def solveFeasible(self, agentNum, u_N, UD):
        session = WolframLanguageSession()
        eps = 1/np.sqrt(agentNum)/agentNum/2.0
        # N[FindInstance[x^2 - 3 y^2 == 1 && 10 < x < 100, {x, y}, Integers]]
        ans = []
        result = []
        while len(result) == 0:
            expr = ""  # expr to evaluate
            for i in range(agentNum-1):
                expr = expr + "x" + str(i) + "> 0.05 &&"
            expr = expr + "x" + str(agentNum-1) + "> 0.05 &&"
            for i in range(agentNum):
                for j in range(i+1, agentNum):
                    # abs(x_i - x_j) <= U_{i,j}
                    expr = expr + \
                        "Abs[x" + str(i) + "-x" + str(j) + "-(" + \
                        str(UD[i, j]) + ")]<=" + str(eps) + "&&"
            for i in range(agentNum-1):
                expr = expr + "x" + str(i) + "+"
            expr = expr + "x" + str(agentNum-1) + "==" + str(u_N) + "&&"
            for i in range(agentNum-1):
                expr = expr + "x" + str(i) + "+"
            expr = expr + "x" + str(agentNum-1) + "<=" + str(u_N)

            expr = expr + ", {"
            for i in range(agentNum-1):
                expr = expr + "x" + str(i) + ","
            expr = expr + "x" + str(agentNum-1) + "}, Reals"

            expr = "N[FindInstance[" + expr + "]]"
            # print(expr)

            result = session.evaluate(wlexpr(expr))
            session.terminate()
            #  print(result)
            if len(result) > 0:
                ans = [result[0][i][1] for i in range(agentNum)]
            eps = eps * 1.1
            print(eps)
        # for i in range(agentNum):
        #     if ans[i] < 0.0000001:
        #         ans[i] = ans[i] + 0.0000001
        print(ans)
        return ans


"""
    GTG
"""


class GTG(ShapleyValue):
    def __init__(self, utility_index):
        super().__init__()
        self.Ut = {}  # t: {}
        self.SV_t = {}  # t: {SV}
        self.utility_index = utility_index
        # TMC paras
        self.Contribution_records = []

        # trunc paras
        self.eps = 0.001
        self.round_trunc_threshold = 0.01

        # converge paras
        self.CONVERGE_MIN_K = 3*10
        self.last_k = 10
        self.CONVERGE_CRITERIA = 0.05

    def compute_shapley_value(self, game, t):
        idxs = game.selected_clients
        N_all = game._n_all
        N = len(idxs)
        self.Contribution_records = []

        # sets = list(powerset(idxs))

        util = {}
        S_0 = ()
        util[S_0] = game.eval_utility(S_0)[self.utility_index]

        # S_all = sets[-1]
        S_all = tuple(idxs)
        util[S_all] = game.eval_utility(S_all)[self.utility_index]

        if abs(util[S_all]-util[S_0]) <= self.round_trunc_threshold:
            sv_dict = {idx: 0 for idx in range(N_all)}
            return sv_dict

        k = 0
        while self.isnotconverge(k):
            for pi in idxs:
                k += 1
                v = [0 for i in range(N+1)]
                v[0] = util[S_0]
                marginal_contribution_k = {idx: 0 for idx in range(N_all)}

                idxs_k = np.concatenate(
                    (np.array([pi]), np.random.permutation([p for p in idxs if p != pi])))

                for j in range(1, N+1):
                    # key = C subset
                    C = idxs_k[:j]
                    C = tuple(np.sort(C, kind='mergesort'))

                    # truncation
                    if abs(util[S_all] - v[j-1]) >= self.eps:
                        if util.get(C) != None:
                            v[j] = util[C]
                        else:
                            v[j] = game.eval_utility(C)[self.utility_index]
                    else:
                        v[j] = v[j-1]

                    # record calculated V(C)
                    util[C] = v[j]

                    # update SV
                    marginal_contribution_k[idxs_k[j-1]] = v[j] - v[j-1]

                self.Contribution_records.append(
                    [marginal_contribution_k[i] for i in range(N_all)])

        # shapley value calculation
        shapley_value = (np.cumsum(self.Contribution_records, 0) /
                         np.reshape(np.arange(1, len(self.Contribution_records)+1), (-1, 1)))[-1:].tolist()[0]

        # store round t results
        self.SV_t[t] = {key: sv for key, sv in enumerate(shapley_value)}

        self.Ut[t] = copy.deepcopy(util)
        # print("Utility index:", self.utility_index)
        # print(self.SV_t[t])
        # print(sum(self.SV_t[t].values()))
        return self.SV_t[t]

    def isnotconverge(self, k):
        if k <= self.CONVERGE_MIN_K:
            return True
        all_vals = (np.cumsum(self.Contribution_records, 0) /
                    np.reshape(np.arange(1, len(self.Contribution_records)+1), (-1, 1)))[-self.last_k:]
        # errors = np.mean(np.abs(all_vals[-last_K:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
        errors = np.mean(np.abs(
            all_vals[-self.last_k:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
        if np.max(errors) > self.CONVERGE_CRITERIA:
            return True
        return False


"""
    MR
"""


class MR(ShapleyValue):
    def __init__(self, utility_index):
        super().__init__()
        self.SV_t = {}  # round t: {id:SV,...}
        self.Ut = {}
        self.utility_index = utility_index

        # for print only
        self.full_set = ()

        # for timer only
        self.st_t = 0

    def compute_shapley_value(self, game, t):
        # timer only
        self.st_t = time.time()

        util = {}
        sets = list(powerset(game.selected_clients))
        for S in tqdm(sets):
            # util[S]=V_S_t(t=t,S=S)
            util[S] = game.eval_utility(S)[self.utility_index]
        # null set
        util[()] = game.eval_utility(())[self.utility_index]

        # for print only
        self.full_set = sets[-1]

        self.SV_t[t] = shapley_value(util, game)

        self.Ut[t] = copy.deepcopy(util)

        # self.print_results(t)

        return self.SV_t[t]


"""
    TMR
"""


class TMR(ShapleyValue):
    def __init__(self, utility_index):
        super().__init__()
        self.SV_t = {}  # round t: {id:SV,...}
        self.Ut = {}
        self.utility_index = utility_index

        # TMR paras
        self.round_trunc_threshold = 0.01

    def compute_shapley_value(self, game, t):
        
        util = {}
        sets = list(powerset(game.selected_clients))

        # TMR round truncation below
        S_0 = ()
        util[S_0] = game.eval_utility(S_0)[self.utility_index]


        S_all = sets[-1]
        util[S_all] = game.eval_utility(S_all)[self.utility_index]

        # TMR round truncation
        if abs(util[S_all]-util[S_0]) <= self.round_trunc_threshold:
            sv_dict = {id: 0 for id in range(game._n_all)}
            return sv_dict

        for S in tqdm(sets):
            util[S] = game.eval_utility(S)[self.utility_index]

        
        self.SV_t[t] = shapley_value(util, game)

        self.Ut[t] = copy.deepcopy(util)

        return self.SV_t[t]

