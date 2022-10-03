import numpy as np
import math
import matplotlib.pyplot as plt

import xmltodict
import pandas as pd
from scipy.spatial import distance_matrix
np.set_printoptions(suppress=True,
                    formatter={'float_kind': '{:.2f}'.format})

from os import listdir
from os.path import isfile, join

import diptest
from sklearn.metrics import pairwise_distances

def read_xml(path, cost_mod, dest_depot=False, source_depot=False):
    """
    Reads data from an xml file 
    """
    with open(path, 'r', encoding='utf-8') as file:
        my_xml = file.read()
    
    my_dict = xmltodict.parse(my_xml)
    
    customer_demand = []
    for request in my_dict['instance']['requests']['request']:
        customer_demand += [math.floor(float(request['quantity']))]
        
    vehicle_capacity = []
    starting_depot = 0
    if type(my_dict['instance']['fleet']['vehicle_profile'])==type([]):
        for vehicle in my_dict['instance']['fleet']['vehicle_profile']:
            vehicle_capacity += [math.floor(float(vehicle['capacity']))]
    if type(my_dict['instance']['fleet']['vehicle_profile'])==type({}):
        vehicle_capacity = [math.floor(float(my_dict['instance']['fleet']['vehicle_profile']['capacity']))]
        starting_depot = my_dict['instance']['fleet']['vehicle_profile']['departure_node']
        
    
    vehicle_capacity = vehicle_capacity * math.ceil(sum(customer_demand)/sum(vehicle_capacity))
    
    
    customer_locations = []
    depot = []
    for customer in my_dict['instance']['network']['nodes']['node']:
        if customer['@type']=="1" and customer['@id'] != starting_depot:
            customer_locations += [[float(customer['cx']),float(customer['cy'])]]
        if customer['@type']=="0" or customer['@id'] == starting_depot:
            depot = [float(customer['cx']),float(customer['cy'])]
    if source_depot:
        customer_locations = [depot]+customer_locations
    if dest_depot:
        customer_locations = customer_locations+[depot]
        
    df = pd.DataFrame(customer_locations, columns=['xcord','ycord'])
    
    distances = distance_matrix(df.values, df.values)

    return customer_demand, vehicle_capacity, distances, customer_locations, depot


def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item
            
def calculate_total_distance(solution, distances):
    """
    Computes the total distance traveled by a solution
    """
    total_distance = 0
    for route in solution:
        for node in route:
            total_distance += distances[node[0]][node[1]]
    return total_distance

def process_solution(solution, num_customers, cluster):
    """
    Converts the solution obtained from the annealer into a more readable format
    Returns an array of tuples indicating the path
    """
    solution_processed = []
    tuples = []
    for i in range(num_customers+2):
        for j in range(num_customers+2):
                tuples += [(i,j)]
    
    for i, val in enumerate(solution.sample.values()):
        if val==1 and i < len(tuples):
            x = tuples[i][0]
            if x==0 or x==num_customers+1:
                x = 0
            else:
                x = cluster[tuples[i][0]-1]+1
            y = tuples[i][1]
            if y==0 or y==num_customers+1:
                y = 0
            else:
                y = cluster[tuples[i][1]-1]+1
            solution_processed += [(x,y)]
    return solution_processed

def compute_cluster_distances(customer_locations, cluster, dest_depot=True):
    """
    Computes the distance matrix for the specified cluster
    """
    inf = 9999999
    locations = [customer_locations[0]] + [customer_locations[i+1] for i in cluster]
    if dest_depot:
        locations += [customer_locations[0]]
    df = pd.DataFrame(locations, columns=['xcord','ycord'])
    cluster_distances = distance_matrix(df.values, df.values)
    for i,line in enumerate(cluster_distances):
        for j,y in enumerate(line):
            cluster_distances[i][0] = inf
            if i == len(cluster_distances)-1:
                cluster_distances[i][j] = inf
            if y == 0.0:
                cluster_distances[i][j] = inf
    return cluster_distances

def compute_tsp_cluster_distances(customer_locations, cluster):
    """
    Computes the distance matrix for the specified cluster
    """
    locations = [customer_locations[0]] + [customer_locations[i+1] for i in cluster]
    df = pd.DataFrame(locations, columns=['xcord','ycord'])
    cluster_distances = distance_matrix(df.values, df.values)
    return cluster_distances

def get_distances_wo_depots(distances):
    """
    Removes the depot from the distance matrix
    """
    distances_wo_depots = []
    for d in distances[1:-1]:
        line = []
        for e in d[1:-1]:
            line += [e]
        distances_wo_depots += [line]
    return distances_wo_depots

def process_tsp_solution(solution, num_customers, cluster):
    tuples = []
    for i in range(num_customers+1):
        for j in range(num_customers+1):
            tuples += [(i,j)]
        
    path = []
    for i, x in enumerate(solution):
        if x==1:
            if tuples[i][1]==0:
                path += [0]
            else:
                path += [cluster[tuples[i][1]-1]+1]
    path += [0]
    solution_processed = []
    for i, elem in enumerate(path[:-1]):
        solution_processed += [(elem, path[i+1])]
        
    return path, solution_processed

def get_files_in_folder(path):
    return [f for f in listdir(path) if isfile(join(path, f))]

def get_cluster_list(clusters, num_customers):
    res_clusters = [-1]*num_customers
    for i,cluster in enumerate(clusters):
        for customer in cluster:
            res_clusters[customer] = i
    return res_clusters

def count_clusters_with_more_demand(clusters, num_clusters, capacity, demand):
    error_clusters = 0
    cluster_demand = [0]*num_clusters
    for i,x in enumerate(clusters):
        if x!=-1:
                cluster_demand[x]+=demand[i]
    for x in cluster_demand:
        if x>capacity:
            error_clusters += 1
    return error_clusters

def plot_data(clusters, num_clusters, capacity, data, depot, demand, solution=None, show_demand=False, 
              plot_numbers=False, show_cluster_demand=False, save_file=None):
        colors =  np.array(np.random.randint(0, 255, size =(num_clusters, 4)))/255
        colors[:,3]=1
        fig, ax = plt.subplots(1,1)
        cluster_demand = [0]*num_clusters
        cluster_cx = [0]*num_clusters
        cluster_cy = [0]*num_clusters
        cluster_count = [0]*num_clusters
        for i,x in enumerate(clusters):
            if x!=-1:
                cluster_demand[x]+=demand[i]
                cluster_cx[x] += data[i][0]
                cluster_cy[x] += data[i][1]
                cluster_count[x] += 1
                
        for i,x in enumerate(cluster_cx):
            cluster_cx[i] /= cluster_count[i]
            cluster_cy[i] /= cluster_count[i]

        for i, x in enumerate(clusters):
            if x == -1:
                plt.scatter(data[i][0], data[i][1], marker="s", s=50, color="black")
                ax.annotate(demand[i], (data[i][0], data[i][1]))
            else:
                if cluster_demand[x]>capacity:
                    marker = "X"
                else:
                    marker = "o"
                plt.scatter(data[i][0], data[i][1], marker=marker, s=50, color=colors[x])
                if solution is not None:
                    for path in solution[x]:
                        if path[0]==0:
                            A = depot
                        else:
                            A = data[path[0]-1]
                        if path[1]==0:
                            B = depot
                        else:
                            B = data[path[1]-1]
                        plt.arrow(A[0], A[1], B[0]-A[0],B[1]-A[1], head_width=1, 
                                length_includes_head=True,color=colors[x])
        
        plt.scatter(depot[0], depot[1], marker="o", s=50, color='black')
        
        if show_cluster_demand:
            for i,x in enumerate(cluster_cx):
                ax.annotate(cluster_demand[i], (cluster_cx[i], cluster_cy[i]))
        
        if show_demand:
            for i, txt in enumerate(demand):
                if clusters[i] != -1:
                    ax.annotate(txt, (data[i][0], data[i][1]))
                
        if plot_numbers:
            for i in range(len(data)):
                ax.annotate(i, (data[i][0], data[i][1]))
                
        if save_file:
            if not save_file.endswith(".png"):
                save_file += ".png"
            plt.savefig(save_file)
        else:
            plt.show()
            
        plt.close()
        
        
def calculate_clusterability(data, plot=True, save_file=None):
    pair_dists = pairwise_distances(data).flatten()
    dip, pval = diptest.diptest(pair_dists)

    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
        plt.title("Dip Score: {:5f}              Dip P Value: {:.5f}".format(dip, pval))
        axes[0].plot([x[0] for x in data], [x[1] for x in data], 'bo')
        axes[1].hist(pair_dists.flatten(), bins=24)
        axes[1].set(xlabel='Dissimilarity', ylabel='Frequency')
            
        fig.tight_layout()

        if save_file:
            if not save_file.endswith(".png"):
                save_file += ".png"
            plt.savefig(save_file)
        else:
            plt.show()

        plt.close()
    return dip, pval

def plot_clustering_parameter_heatmaps(problem_name, csv_file, save_file=None):
    if type(csv_file) == str:
        df = pd.read_csv(csv_file)

        constraint_1 = df['Constraint 1'].unique()
        constraint_2 = df['Constraint 2'].unique()
        len_constraint_2 = len(constraint_2)
        len_constraint_1 = len(constraint_1)

        node_errors = np.array(df['Unassigned Nodes'].to_list()).reshape((len_constraint_1, len_constraint_2))
        demand_errors = np.array(df['Demand Errors'].to_list()).reshape((len_constraint_1, len_constraint_2))
        silhouette = np.array(df['Silhouette'].to_list()).reshape((len_constraint_1, len_constraint_2))

        rsilhouette = np.around(silhouette, 2)
    else:
        all_node_errors = []
        all_demand_errors = []
        all_silhouette = []
        for csv_f in csv_file:
            df = pd.read_csv(csv_f)
            
            constraint_1 = df['Constraint 1'].unique()
            constraint_2 = df['Constraint 2'].unique()
            len_constraint_2 = len(constraint_2)
            len_constraint_1 = len(constraint_1)
            
            norm_node_errors = (df['Unassigned Nodes']-df['Unassigned Nodes'].min())/(df['Unassigned Nodes'].max()-df['Unassigned Nodes'].min())
            if df['Demand Errors'].max() != 0:
                norm_demand_errors = (df['Demand Errors']-df['Demand Errors'].min())/(df['Demand Errors'].max()-df['Demand Errors'].min())
            else:
                norm_demand_errors = df['Demand Errors']
            norm_silhouette = (df['Silhouette']-df['Silhouette'].min())/(df['Silhouette'].max()-df['Silhouette'].min())
            node_errors = np.around(np.array(norm_node_errors.to_list()),2).reshape((len_constraint_1, len_constraint_2))
            demand_errors = np.around(np.array(norm_demand_errors.to_list()),2).reshape((len_constraint_1, len_constraint_2))
            silhouette = np.around(np.array(norm_silhouette.to_list()),2).reshape((len_constraint_1, len_constraint_2))
            
            all_node_errors += [node_errors]
            all_demand_errors += [demand_errors]
            all_silhouette += [silhouette]
        
        node_errors = np.around(np.average(np.array(all_node_errors), axis=0), 2)
        demand_errors = np.around(np.average(np.array(all_demand_errors), axis=0), 2)
        rsilhouette = np.around(np.average(np.array(all_silhouette), axis=0),2)
        
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,4))

    # Unassigned Node Errors
    im = ax[0].imshow(node_errors)
    # Show all ticks and label them with the respective list entries
    ax[0].set_xticks(np.arange(len(constraint_2)))
    ax[0].set_xticklabels(constraint_2)
    ax[0].set_yticks(np.arange(len(constraint_1)))
    ax[0].set_yticklabels(constraint_1)
    # Loop over data dimensions and create text annotations.
    for i in range(len(constraint_1)):
        for j in range(len(constraint_2)):
            text = ax[0].text(j, i, node_errors[i, j],
                        ha="center", va="center", color="w")

    ax[0].set_title("Unassigned Node Errors")
    ax[0].set_ylabel("Constraint 1 Multiplier")
    ax[1].set_xlabel("Constraint 2 Multiplier")

    # Cluster Demand Errors
    im = ax[1].imshow(demand_errors)
    # Show all ticks and label them with the respective list entries
    ax[1].set_xticks(np.arange(len(constraint_2)))
    ax[1].set_xticklabels(constraint_2)
    ax[1].set_yticks(np.arange(len(constraint_1)))
    ax[1].set_yticklabels(constraint_1)
    # Loop over data dimensions and create text annotations.
    for i in range(len(constraint_1)):
        for j in range(len(constraint_2)):
            text = ax[1].text(j, i, demand_errors[i, j],
                        ha="center", va="center", color="w")

    ax[1].set_title("Cluster Demand Errors")

    # Silhouette Score
    im = ax[2].imshow(rsilhouette)
    # Show all ticks and label them with the respective list entries
    ax[2].set_xticks(np.arange(len(constraint_2)))
    ax[2].set_xticklabels(constraint_2)
    ax[2].set_yticks(np.arange(len(constraint_1)))
    ax[2].set_yticklabels(constraint_1)
    # Loop over data dimensions and create text annotations.
    for i in range(len(constraint_1)):
        for j in range(len(constraint_2)):
            text = ax[2].text(j, i, rsilhouette[i, j],
                        ha="center", va="center", color="w")

    ax[2].set_title("Silhouette Score")

    fig.tight_layout()
    fig.suptitle("{} Constraint Multiplier Heatmaps".format(problem_name))
    
    if save_file:
        if not save_file.endswith(".png"):
            save_file += ".png"
        plt.savefig(save_file)
    else:
        plt.show()

    plt.close()
    
def plot_routing_parameter_heatmaps(problem_name, csv_file, save_file=None):
    if type(csv_file) == str:
        df = pd.read_csv(csv_file)

        constraint_1 = df['Constraint 1'].unique()
        constraint_2 = df['Constraint 2'].unique()
        len_constraint_2 = len(constraint_2)
        len_constraint_1 = len(constraint_1)

        errors = np.array(df['Errors'].to_list()).reshape((len_constraint_1, len_constraint_2))
        distance = np.array(df['Distance'].to_list()).reshape((len_constraint_1, len_constraint_2))

        rdistance = np.around(distance, 2)
    else:
        all_errors = []
        all_distance = []
        for csv_f in csv_file:
            df = pd.read_csv(csv_f)
            
            constraint_1 = df['Constraint 1'].unique()
            constraint_2 = df['Constraint 2'].unique()
            len_constraint_2 = len(constraint_2)
            len_constraint_1 = len(constraint_1)
            
            if df['Errors'].max() != 0:
                norm_errors = (df['Errors']-df['Errors'].min())/(df['Errors'].max()-df['Errors'].min())
            else:
                norm_errors = df['Errors']
            norm_distance = (df['Distance']-df['Distance'].min())/(df['Distance'].max()-df['Distance'].min())

            errors = np.around(np.array(norm_errors.to_list()),2).reshape((len_constraint_1, len_constraint_2))
            distance = np.around(np.array(norm_distance.to_list()),2).reshape((len_constraint_1, len_constraint_2))

            all_errors += [errors]
            all_distance += [distance]
        
        errors = np.around(np.average(np.array(all_errors), axis=0), 2)
        rdistance = np.around(np.average(np.array(all_distance), axis=0),2)
        
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    # Number of Errors
    im = ax[0].imshow(errors)
    # Show all ticks and label them with the respective list entries
    ax[0].set_xticks(np.arange(len(constraint_2)))
    ax[0].set_xticklabels(constraint_2)
    ax[0].set_yticks(np.arange(len(constraint_1)))
    ax[0].set_yticklabels(constraint_1)
    # Loop over data dimensions and create text annotations.
    for i in range(len(constraint_1)):
        for j in range(len(constraint_2)):
            text = ax[0].text(j, i, errors[i, j],
                        ha="center", va="center", color="w")

    ax[0].set_title("Number of Errors")
    ax[0].set_ylabel("Constraint A Multiplier")
    ax[0].set_xlabel("Constraint B Multiplier")

    # Total Distance
    im = ax[1].imshow(rdistance)
    # Show all ticks and label them with the respective list entries
    ax[1].set_xticks(np.arange(len(constraint_2)))
    ax[1].set_xticklabels(constraint_2)
    ax[1].set_yticks(np.arange(len(constraint_1)))
    ax[1].set_yticklabels(constraint_1)
    # Loop over data dimensions and create text annotations.
    for i in range(len(constraint_1)):
        for j in range(len(constraint_2)):
            text = ax[1].text(j, i, rdistance[i, j],
                        ha="center", va="center", color="w")

    ax[1].set_title("Total Distance")
    ax[1].set_xlabel("Constraint B Multiplier")
    ax[1].set_ylabel("Constraint A Multiplier")

    fig.tight_layout()
    fig.suptitle("{} Constraint Multiplier Heatmaps".format(problem_name))
    plt.subplots_adjust(top=0.85)
    
    if save_file:
        if not save_file.endswith(".png"):
            save_file += ".png"
        plt.savefig(save_file)
    else:
        plt.show()

    plt.close()