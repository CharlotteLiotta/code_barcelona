import numpy as np # type: ignore
import scipy # type: ignore
import matplotlib.pyplot as plt # type: ignore
import random
from numba import njit, jit
from copy import deepcopy
from numba import njit, prange

### URBAN ECONOMICS

def compute_utility_manually(Y, T, q, R, BETA, OPTION_HEALTH = 0, N = 0, vkm = 0, marginal_cost_pollution = 0):
    if OPTION_HEALTH == 0:
        u = (Y - T - q * R) ** (1 - BETA) * q ** BETA
    elif OPTION_HEALTH == 1:
        health = vkm * marginal_cost_pollution / N
        u = (Y - T - q * R - health) ** (1 - BETA) * q ** BETA
    return u


def compute_error_in_population(u, N, BETA, Y, transport_cost, B, KAPPA, RHO, L, housing_lag = None):
    '''
    Compute the difference between the population estimated by the model if 
    the utility is equal to u and the actual population.

        Parameters:
            N (float): Actual population
            u (float): Utility
                        
        Returns:
            error_population (float): Difference between the estimated and actual population
        '''

    R = compute_rents(BETA, Y, u, transport_cost)
    q = compute_dwelling_size(BETA, Y, transport_cost, R)
    if housing_lag is None:
        n = compute_population(B, KAPPA, R, RHO, L, q)
    else:
        n = housing_lag / q
    #print("Estimated_population", np.nansum(n))
    #print("Error", N - np.nansum(n))
    error_population = np.abs(N - np.nansum(n))
    return error_population


def compute_rents(beta, Y, u, T):
    '''
    Compute the bid-rent at each location in the city.

            Parameters:
                    beta (float): Parameter of the utility function
                    Y (float): Average income
                    u (float): Average utility
                    T (array): Transportation costs at each location

            Returns:
                    R (array): Bid-rent at each location
    '''
    alpha = 1-beta
    Ro = (((alpha ** alpha) * (beta ** beta) * Y) / u) ** (1/beta)
    R = Ro * ((1 - (T/Y)) ** (1/beta))
    return R


def compute_dwelling_size(beta, Y, T, R):
    '''
    Compute the dwelling size at each location in the city.

            Parameters:
                    beta (float): Parameter of the utility function
                    Y (float): Average income
                    T (array): Transportation costs at each location
                    R (array): Rents at each location

            Returns:
                    q (array): Dwelling size at each location
    '''

    q = beta * (Y - T) / R
    return q

def compute_population(b, kappa, R, rho, L, q):
    '''
    Compute the population at each location in the city.

            Parameters:
                    b (float): Parameter of the housing supply function
                    kappa (float): Parameter of the housing supply function
                    R (array): Rents at each location
                    rho (array): Interest rate
                    L (array): Land available for housing at each location
                    q (array): Dwelling size at each location

            Returns:
                    q (array): Dwelling size at each location
    '''

    a = 1-b
    n = kappa ** (1/a) * (b * R / rho) ** (b/a) * L / q
    return n

def compute_transport_cost(COST_CAR, COST_PT, tax):
    return np.fmin(COST_CAR * tax, COST_PT), np.argmin([COST_CAR * tax, COST_PT], 0)

### OUTPUT FROM THE URBAN ECON MODEL

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array + 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))


def compute_change_in_welfare(utility_without_tax, utility_with_tax):
    """ Compute the impact of the change in utility on welfare"""

    relative_change_utility = (100 * (utility_with_tax - utility_without_tax)) / utility_without_tax
    return (1 / (1 + np.exp(-0.15 * relative_change_utility)))

def compute_change_in_inequalities(utility_without_tax, utility_with_tax):
    if gini(np.array(utility_without_tax)) != gini(np.array(utility_with_tax)):
        Q = (1 / (1 + np.exp(10 * (gini(np.array(utility_with_tax)) - gini(np.array(utility_without_tax)))))) #/ ))) #gini(np.array(utility_without_tax)))))
    else:
        Q = 0.5
    return Q

def compute_change_in_emissions(n_with_tax, distance, transport_mode, emissions_init):
    relative_change_emission = 100 * (sum((n_with_tax * distance)[transport_mode == 0]) - emissions_init) / emissions_init

    return (1 / (1 + np.exp(0.02 * relative_change_emission)))

### POLICY SUPPORT FUNCTIONS

def compute_political_opinion(score_welfare, score_ineq, score_emissions, I, BETA_OPINION, opinion_distance_matrix):
    return BETA_OPINION[6] + (BETA_OPINION[0] + BETA_OPINION[3] * I) * score_welfare + (BETA_OPINION[1] + BETA_OPINION[4] * I) * score_ineq + (BETA_OPINION[2] + BETA_OPINION[5] * I) * score_emissions

@njit(parallel = True)
def compute_social_interactions(political_opinion, I, GAMMA):
    N = len(I)
    result = np.zeros(N)
    
    for i in prange(N):
        #print(100 * i/N, "%")
        abs_diff = np.abs(I - I[i])
        matrix_opinion_row = np.exp(-abs_diff)
        
        weighted_sum = np.sum(political_opinion * matrix_opinion_row)
        sum_matrix_opinion = np.sum(matrix_opinion_row)
        
        result[i] = ((1 - GAMMA) * political_opinion[i]) + (GAMMA * (weighted_sum / sum_matrix_opinion))
    
    return result

### ABM

@njit
def compute_indiv_distance_matrix(N, distance, n_with_tax):
    opinion_distance_matrix = np.zeros((N, len(distance)))

    step = 0
    for k in range(len(distance)):
        nb_pers = round(n_with_tax[k])
        if step+nb_pers < N:
            opinion_distance_matrix[step:step+nb_pers, k] = np.ones(nb_pers)
            step += nb_pers
        else:
            opinion_distance_matrix[step:N, k] = np.ones(N-step)
            break
    return opinion_distance_matrix

@njit
def update_indiv_distance_matrix(indiv_distance_matrix, n):
    #print(np.nansum(indiv_distance_matrix,0) - n)
    indiv_distance_matrix_new = (indiv_distance_matrix)

    ### first iteration
    lag = sum(np.abs(np.nansum(indiv_distance_matrix_new,0) - n))
    #print(sum(np.abs(np.nansum(indiv_distance_matrix_new,0) - n)))
    destination = np.argmin((np.nansum(indiv_distance_matrix_new,0) - n))
    origin = np.argmax((np.nansum(indiv_distance_matrix_new,0) - n))
    #indiv_moving = (np.argwhere(indiv_distance_matrix_new[:, origin] == 1)[np.random.randint(1,len(np.argwhere(indiv_distance_matrix_new[:, origin] == 1)))]).astype(int)[0]
    indiv_moving = np.random.choice(np.argwhere(indiv_distance_matrix_new[:, origin] == 1).flatten())
    if indiv_distance_matrix_new[indiv_moving, origin] != 1:
        raise Exception
    indiv_distance_matrix_new[indiv_moving, origin] = 0
    if indiv_distance_matrix_new[indiv_moving, destination] != 0:
        raise Exception
    indiv_distance_matrix_new[indiv_moving, destination] = 1
    new = sum(np.abs(np.nansum(indiv_distance_matrix_new,0) - n))
    #loop
    while np.abs(new-lag)> 0.01:
        lag = new
        #print(sum(np.abs(np.nansum(indiv_distance_matrix_new,0) - n)))
        destination = np.argmin((np.nansum(indiv_distance_matrix_new,0) - n))
        origin = np.argmax((np.nansum(indiv_distance_matrix_new,0) - n))
        #indiv_moving = (np.argwhere(indiv_distance_matrix_new[:, origin] == 1)[np.random.randint(1,len(np.argwhere(indiv_distance_matrix_new[:, origin] == 1)))]).astype(int)[0]
        indiv_moving = np.random.choice(np.argwhere(indiv_distance_matrix_new[:, origin] == 1).flatten())
        if indiv_distance_matrix_new[indiv_moving, origin] != 1:
            raise Exception
        indiv_distance_matrix_new[indiv_moving, origin] = 0
        if indiv_distance_matrix_new[indiv_moving, destination] != 0:
            raise Exception
        indiv_distance_matrix_new[indiv_moving, destination] = 1
        new = sum(np.abs(np.nansum(indiv_distance_matrix_new,0) - n))

    return indiv_distance_matrix_new

@njit
def compute_proba_of_moving(housing_lag, housing_without_inertia):
    """ Compute the probability of moving from, and moving to, each spatial unit"""

    proba_of_moving_from = np.zeros(len(housing_lag))
    mask = housing_lag > 0
    proba_of_moving_from[mask] = (housing_lag[mask] - housing_without_inertia[mask]) / housing_lag[mask]
    
    proba_of_moving_to = (housing_without_inertia - housing_lag)
    proba_of_moving_to[proba_of_moving_to < 0] = 0
    proba_of_moving_to /= np.nansum(proba_of_moving_to)
    return proba_of_moving_from, proba_of_moving_to


@njit
def make_people_move(indiv_distance_matrix_new, N, distance, indiv_distance_matrix, proba_of_moving_from, proba_of_moving_to):
    
    has_moved = np.zeros(N)
    indiv_moving = np.random.binomial(1, 0.3, N) #Each individual has a 30% chance to be willing to move.

    for i in np.arange(N):
        if indiv_moving[i] == 1:
            if sum(indiv_distance_matrix[i,:]) > 0:
                proba_of_moving_here = sum((indiv_distance_matrix[i,:] * proba_of_moving_from)[indiv_distance_matrix[i,:] > 0])
                if proba_of_moving_here < 0:
                    proba_of_moving_here = 0
                moving = np.random.binomial(1, proba_of_moving_here)
                if moving == 1:
                    has_moved[i] = 1
                    indiv_distance_matrix_new[i,:] = np.zeros(len(distance))
                    #destination = np.random.choice(np.arange(len(distance)), p=proba_of_moving_to)
                    destination = np.arange(len(distance))[np.searchsorted(np.cumsum(proba_of_moving_to), np.random.random(), side="right")]
                    indiv_distance_matrix_new[i,destination] = 1

    return indiv_distance_matrix_new, has_moved

### PLOT AND VISUALIZE VARIABLES

def plot_variables(distance, variable, title):
    # Normalize the values to range [0, 1] for color mapping
    norm = plt.Normalize(variable[~np.isnan(variable)].min(), variable[~np.isnan(variable)].max())

    # Create a color map (Red to Green)
    cmap = plt.cm.RdYlGn

    # Plot the circles
    fig, ax = plt.subplots()
    for (xi, vi) in zip(distance[::-1], variable[::-1]):
        color = cmap(norm(vi))
        circle = plt.Circle((0, 0), xi, color=color, alpha=0.7)
        ax.add_patch(circle)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='vertical', label='')

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    # Set limits to ensure all circles are visible
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)

    plt.axis('off')

    plt.title(title)

    plt.show()

def compute_weighted_mean_opinions(var, opinion_distance_matrix, N):
    weighted_mean_opinion = np.zeros(opinion_distance_matrix.shape[1])
    weighted_mean_opinion[np.nansum(opinion_distance_matrix, 0) > 0] = np.nansum(opinion_distance_matrix * var.reshape(N, 1), 0)[np.nansum(opinion_distance_matrix, 0) > 0] / np.nansum(opinion_distance_matrix, 0)[np.nansum(opinion_distance_matrix, 0) > 0]
    weighted_mean_opinion[np.nansum(opinion_distance_matrix, 0) == 0] = np.nan
    return weighted_mean_opinion

def plot_tax_suppport(save_tax, save_median_support):

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (year)')
    ax1.set_ylabel('tax', color=color)
    ax1.plot(save_tax[1:], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('median support', color=color)  # we already handled the x-label with ax1
    ax2.plot(save_median_support[1:], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()