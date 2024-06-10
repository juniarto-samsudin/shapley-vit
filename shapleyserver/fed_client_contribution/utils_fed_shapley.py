import copy
import numpy as np
import time
from tqdm import tqdm
from federated_learning.utils import evaluation, evaluation_group_fairness, evaluation_statistical_parity, get_aggregated_model
from fed_client_contribution.utils_shapley import (
    powerset,
    ncr,
)






def utility(args, previous_utility, previous_global_model, fake_server, clients):
    # fake_server.model_agg2([client.model_train for client in clients], selected_clients=clients)
    fake_server.model_agg3(previous_global_model.global_model, [
                           client.model_train for client in clients], selected_clients=clients)
    fake_test_acc, fake_test_loss = evaluation(
        args, fake_server.global_model, fake_server.valid_loader)
    # group_acc_diff, group_loss_diff = evaluation_group_fairness(args, fake_server.global_model, fake_server.group_testloader)
    # group_positive_prediction = evaluation_statistical_parity(args, fake_server.global_model, fake_server.group_testloader)
    # return fake_test_acc, fake_test_loss, group_acc_diff, group_loss_diff, group_positive_prediction
    return fake_test_acc, fake_test_loss


# baseline
def compute_shapley_value_baseline(args, utilities_dict, idxs_users):
    N = len(idxs_users)
    roundly_valuation_baseline = np.zeros(args.num_clients)
    for i in range(len(idxs_users)):
        tmp_indices = list(idxs_users)
        current_i = tmp_indices.pop(i)
        subpowerset = powerset(tmp_indices)
        val = 0
        for s in subpowerset.keys():
            si = tuple(sorted(list(s)+[current_i]))
            val += (utilities_dict[si]-utilities_dict[s]) / ncr(N - 1, len(s))
        roundly_valuation_baseline[current_i] = val / N
    return roundly_valuation_baseline


# ground truth
def compute_shapley_value_groundtruth(args, utilities_dict):
    N = args.num_users
    roundly_valuation_groundtruth = np.zeros(N)
    for i in range(N):
        tmp_indices = list(range(N))
        current_i = tmp_indices.pop(i)
        subpowerset = powerset(tmp_indices)
        val = 0
        for s in subpowerset.keys():
            si = tuple(sorted(list(s)+[current_i]))
            val += (utilities_dict[si]-utilities_dict[s]) / ncr(N - 1, len(s))
        roundly_valuation_groundtruth[current_i] = val / N
    return roundly_valuation_groundtruth


# mask in each round
def roundly_mask(idxs_users, all_subsets):
    mask_vec = np.zeros(len(all_subsets))
    subpowerset = powerset(idxs_users)
    for s in subpowerset.keys():
        i = all_subsets[s]
        mask_vec[i] = 1
    return mask_vec


# ComFedSV
def compute_shapley_value_from_matrix(args, utility_matrix, all_subsets):
    T = args.epochs
    N = args.num_users
    valuation_completed = np.zeros(N)
    for i in range(N):
        sublist = list(range(N))
        sublist.pop(i)
        subpowerset = powerset(sublist)
        for s in subpowerset.keys():
            # id1 = all_subsets.index(s)
            id1 = all_subsets[s]
            id2 = all_subsets[tuple(sorted(list(s)+[i]))]
            for t in range(T):
                v1 = utility_matrix[t, id1]
                v2 = utility_matrix[t, id2]
                val = (v2 - v1) / ncr(N - 1, len(s))
                # summation of Shapley values of all epochs
                valuation_completed[i] += val
        valuation_completed[i] /= N
    return valuation_completed


def compute_utilities(args,
                      previous_utility,
                      previous_global_model,
                      fake_model,
                      all_subsets,
                      clients,
                      idxs_users,
                      utility_dim,
                      shapley_non_participating_clients):
    """
        Multi-objective utility function
    """
    # accuracy, loss
    utilities = []
    utilities_dict = []
    for i in range(utility_dim):
        utilities.append(np.zeros(len(all_subsets)))
        utilities_dict.append({})

    if (shapley_non_participating_clients):
        # deep copy clients model
        clients_copy = [copy.deepcopy(client) for client in clients]

        # for non-participating clients, sync with the previous global model
        for i in range(args.num_clients):
            if i not in idxs_users:
                clients_copy[i].sync_with_server(previous_global_model)

        for _, indices in enumerate(tqdm(powerset(range(args.num_clients)).keys())):
            selected_clients = [clients_copy[i] for i in indices]
            u = utility(args, previous_utility, previous_global_model,
                        fake_model, selected_clients)
            # u = utility_fairness(args, None, fake_model, weights, group_test_dataset)

            # store values
            for i in range(utility_dim):
                utilities[i][all_subsets[indices]] = u[i]
                utilities_dict[i][indices] = u[i]

    else:
        for _, indices in enumerate(tqdm(powerset(idxs_users).keys())):
            selected_clients = [clients[i] for i in indices]
            u = utility(args, previous_utility, previous_global_model,
                        fake_model, selected_clients)
            # u = utility_fairness(args, None, fake_model, weights, group_test_dataset)

            # store values
            for i in range(utility_dim):
                utilities[i][all_subsets[indices]] = u[i]
                utilities_dict[i][indices] = u[i]
    return utilities, utilities_dict


def compute_utilities_lazy(args,
                           previous_utility,
                           client_model_all_rounds,
                           client_model_selection_matrix,
                           fake_server,
                           clients_all,
                           init_global_model,
                           all_subsets,
                           utility_dim,
                           current_round,
                           include_from_round
                           ):
    utilities = []
    utilities_dict = []

    for i in range(utility_dim):
        utilities.append(np.zeros(len(all_subsets)))
        utilities_dict.append({})


    for _, indices in enumerate(tqdm(powerset(range(args.num_clients)).keys(), disable=False)):
        # model reconstruction
        per_round_aggregated_models = []
        for i in range(current_round + 1):
            if (i < include_from_round):
                    continue
            # print(i)
            participating_indices = [
                j for j in indices if client_model_selection_matrix[i][j]]
            if (len(participating_indices) > 0):
                # print(i, indices, participating_indices)
                per_round_aggregated_models.append(
                    get_aggregated_model([client_model_all_rounds[i][j] for j in participating_indices],
                                         fake_server.get_agg_ratio(selected_clients=[clients_all[j] for j in participating_indices]))
                )
        
        # print()
        fake_server.model_agg_lazy(
            init_global_model, per_round_aggregated_models)
        fake_test_acc, fake_test_loss = evaluation(
            args, fake_server.global_model, fake_server.valid_loader)

        # acc
        utilities[0][all_subsets[indices]] = fake_test_acc - previous_utility[0]
        utilities_dict[0][indices] = fake_test_acc - previous_utility[0]

        # loss
        utilities[1][all_subsets[indices]] = fake_test_loss - previous_utility[1]
        utilities_dict[1][indices] = fake_test_loss - previous_utility[1]
        
    return utilities, utilities_dict




def compute_shapley_value_for_participating_clients(args, 
                                                    utilities_dict_list, 
                                                    mask, 
                                                    shapley_non_participating_clients
                                                ):
    T = args.rounds
    valuation_per_round = []
    for t in range(T):
        if (not shapley_non_participating_clients):
            participating_clients = np.where(mask[:, :args.num_clients][t] == 1)[
                0]  # participating clients in epoch t
        else:
            participating_clients = np.arange(args.num_clients)  # all clients
        valuation_completed = compute_shapley_corrected(
            utilities_dict_list[t], participating_clients.tolist())
        valuation_per_round.append(valuation_completed)

    return valuation_per_round


def compute_shapley_value_lazy_approach(args, 
                                utilities_dict_list
                            ):
    valuation_per_round = []
    for t in range(len(utilities_dict_list)):
        participating_clients = np.arange(args.num_clients)  # all clients
        valuation_completed = compute_shapley_corrected(
            utilities_dict_list[t], participating_clients.tolist())
        valuation_per_round.append(valuation_completed)
    return valuation_per_round


def print_shapley_value(utility_map,
                        utilities_dict,
                        participating_clients,
                        logger
                    ):
    for key in utility_map:
        shapley_values = compute_shapley_corrected(
                utilities_dict[key], participating_clients)
        logger.info(
            f"==== Shapley values for {utility_map[key]} ====")
        if (True):
            print()
            from pprint import pprint, pformat
            logger.info(
                f"utility dict\n{pformat(utilities_dict[key])}")
            logger.info("")
            logger.info(
                f"Shapley value\n{pformat(shapley_values)}")
            logger.info("")

def get_selection_dict(num_clients, idxs_participating_clients):
    selection_dict = {}
    for i in range(num_clients):
        selection_dict[i] = False
    for i in idxs_participating_clients:
        selection_dict[i] = True
    return selection_dict


def get_optimal_subset(server,
                       clients,
                       utilities_dict,
                       idxs_participating_clients):

    idxs_best_subset = min(utilities_dict, key=utilities_dict.get)
    num_clients = len(clients)

    # for non-participating clients, sync with the previous global model. Use deepcopy so that non-participating clients does not get the updated model
    clients_all_copy = [copy.deepcopy(client) for client in clients]
    for i in range(num_clients):
        if i not in idxs_participating_clients:
            clients_all_copy[i].sync_with_server(server)
    selected_clients = [clients_all_copy[i]
                    for i in idxs_best_subset]
    
    return selected_clients, idxs_best_subset


def get_optimal_subset_multi_objectives(server,
                                        clients,
                                        utilities_dict_list,
                                        idxs_participating_clients):
    # accuracy is already bounded between 0 and 1
    # since loss is not bounded, we normalize it between 0 and 1
    max_loss = utilities_dict_list[1][-1][max(
        utilities_dict_list[1][-1], key=utilities_dict_list[1][-1].get)]
    min_loss = utilities_dict_list[1][-1][min(
        utilities_dict_list[1][-1], key=utilities_dict_list[1][-1].get)]
    max_acc = utilities_dict_list[0][-1][max(
        utilities_dict_list[0][-1], key=utilities_dict_list[0][-1].get)]
    min_acc = utilities_dict_list[0][-1][min(
        utilities_dict_list[0][-1], key=utilities_dict_list[0][-1].get)]

    # print(max_loss, min_loss, max_acc, min_acc)

    combined_utility_clients = {}
    for key in utilities_dict_list[0][-1]:

        combined_utility_clients[key] = 0

        # acc
        if (max_acc == min_acc):
            combined_utility_clients[key] += 1
        else:
            combined_utility_clients[key] += (
                utilities_dict_list[0][-1][key] - min_acc) / (max_acc - min_acc)

        # loss
        if (max_loss == min_loss):
            combined_utility_clients[key] -= 1
        else:
            combined_utility_clients[key] -= (
                utilities_dict_list[1][-1][key] - min_loss) / (max_loss - min_loss)

    idxs_best_subset = max(
        combined_utility_clients, key=combined_utility_clients.get)

    num_clients = len(clients)
    # for non-participating clients, sync with the previous global model. Use deepcopy so that non-participating clients does not get the updated model
    clients_all_copy = [copy.deepcopy(
        client) for client in clients]
    for i in range(num_clients):
        if i not in idxs_participating_clients:
            clients_all_copy[i].sync_with_server(server)
    selected_clients = [clients_all_copy[i]
                        for i in idxs_best_subset]
    

    return selected_clients, idxs_best_subset
