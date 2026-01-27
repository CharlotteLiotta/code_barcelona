import numpy as np # type: ignore
from numba import njit, prange # type: ignore

@njit
def compute_indiv_loc_matrix(N, len_gdf, n):
    """ Build the agents location matrix.
    
    Allocate the N agents to
    the len_gdf census tracts of analysis,
    based on the simlulated population density n 
    """
    
    opinion_distance_matrix = np.zeros((N, len_gdf))
    step = 0

    for k in range(len_gdf):
        nb_pers = int(round(n[k]))  # rounding outside of array slices is OK in njit

        end = step + nb_pers
        if end < N:
            for i in range(step, end):
                opinion_distance_matrix[i, k] = 1.0
            step = end
        else:
            for i in range(step, N):
                opinion_distance_matrix[i, k] = 1.0
            break

    missing_ppl = int(N - opinion_distance_matrix.sum())

    if missing_ppl > 0:
        print("missing ppl", missing_ppl)
    
        # Get location indices sorted by population in descending order
        pop_per_location = opinion_distance_matrix.sum(axis=0)
        sorted_locs = np.argsort(-pop_per_location)  # descending order

        # Start assigning one agent per location
        step = int(N - missing_ppl)
        for i in range(missing_ppl):
            opinion_distance_matrix[step + i, sorted_locs[i % len(sorted_locs)]] = 1.0

    return opinion_distance_matrix

@njit
def compute_indiv_loc_matrix2(N, len_gdf, n):

    alloc = np.empty(len_gdf, dtype=np.int64)
    remainders = np.empty(len_gdf)

    # 1. Floor allocation
    total = 0
    for k in range(len_gdf):
        alloc[k] = int(np.floor(n[k]))
        total += alloc[k]
        remainders[k] = n[k] - alloc[k]

    # 2. Distribute remaining agents
    remaining = N - total
    if remaining > 0:
        idx = np.argsort(remainders)[::-1]  # descending remainders
        for i in range(remaining):
            alloc[idx[i]] += 1

    # 3. Build matrix
    opinion_distance_matrix = np.zeros((N, len_gdf))
    step = 0
    for k in range(len_gdf):
        for i in range(step, step + alloc[k]):
            opinion_distance_matrix[i, k] = 1.0
        step += alloc[k]

    return opinion_distance_matrix


@njit
def compute_proba_of_moving(housing_lag, housing_without_inertia):
    """ Compute the probability of moving from, and moving to, each spatial unit"""
    
    len_housing = len(housing_lag)
    proba_of_moving_from = np.zeros(len_housing)
    proba_of_moving_to = np.zeros(len_housing)

    total_positive_change = 0.0

    for i in range(len_housing):
        if housing_lag[i] > 0:
            diff = housing_lag[i] - housing_without_inertia[i]
            proba_of_moving_from[i] = diff / housing_lag[i]

        delta = housing_without_inertia[i] - housing_lag[i]
        if delta > 0:
            proba_of_moving_to[i] = delta
            total_positive_change += delta

    if total_positive_change > 0:
        for i in range(len_housing):
            proba_of_moving_to[i] /= total_positive_change

    return proba_of_moving_from, proba_of_moving_to

#@njit
def make_people_move(indiv_loc_matrix_new, N, len_gdf, indiv_loc_matrix, proba_of_moving_from, proba_of_moving_to, PROBA_MOVE):
    
    has_moved = np.zeros(N)
    indiv_moving = np.random.binomial(1, PROBA_MOVE, N) #Each individual has a 30% chance to be willing to move.

    for i in np.arange(N):
        if indiv_moving[i] == 1:
            if sum(indiv_loc_matrix[i,:]) > 0:
                proba_of_moving_here = sum((indiv_loc_matrix[i,:] * proba_of_moving_from)[indiv_loc_matrix[i,:] > 0])
                if proba_of_moving_here < 0:
                    proba_of_moving_here = 0
                moving = np.random.binomial(1, proba_of_moving_here)
                if moving == 1:
                    has_moved[i] = 1
                    indiv_loc_matrix_new[i,:] = np.zeros(len_gdf)
                    #destination = np.random.choice(np.arange(len(distance)), p=proba_of_moving_to)
                    destination = np.arange(len_gdf)[np.searchsorted(np.cumsum(proba_of_moving_to), np.random.random(), side="right")]
                    indiv_loc_matrix_new[i,destination] = 1

    return indiv_loc_matrix_new, has_moved

@njit
def make_people_move(indiv_loc_matrix_new, N, len_gdf, indiv_loc_matrix, proba_of_moving_from, proba_of_moving_to, PROBA_MOVE):
    
    has_moved = np.zeros(N)
    indiv_moving = np.random.binomial(1, PROBA_MOVE, N) #Each individual has a 30% chance to be willing to move.

    for i in np.arange(N):
        if indiv_moving[i] == 1:
            if sum(indiv_loc_matrix[i,:]) > 0:
                proba_of_moving_here = sum((indiv_loc_matrix[i,:] * proba_of_moving_from)[indiv_loc_matrix[i,:] > 0])
                if proba_of_moving_here < 0:
                    proba_of_moving_here = 0
                moving = np.random.binomial(1, proba_of_moving_here)
                if moving == 1:
                    has_moved[i] = 1
                    indiv_loc_matrix_new[i,:] = np.zeros(len_gdf)
                    #destination = np.random.choice(np.arange(len(distance)), p=proba_of_moving_to)
                    destination = np.arange(len_gdf)[np.searchsorted(np.cumsum(proba_of_moving_to), np.random.random(), side="right")]
                    indiv_loc_matrix_new[i,destination] = 1

    return indiv_loc_matrix_new, has_moved