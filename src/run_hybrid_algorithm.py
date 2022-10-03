from hybrid.clustering.QUBO_Clustering import QUBO_Clustering
from solver.Google_OR_Solver import solve_tsp

from helpers.utils import read_xml, plot_data, count_clusters_with_more_demand, compute_tsp_cluster_distances, calculate_total_distance
from helpers.check_solution import check_solution

from sklearn.metrics import silhouette_score

import multiprocessing as mp


def run_qubo_clustering(distances, vehicle_capacity, customer_demand, customer_locations, depot,
                        save_file, num_reads, constraint_1_multiplier, constraint_2_multiplier):
    
      
    # QUBO Clustering Algorithm
    qubo_clustering = QUBO_Clustering(distances, vehicle_capacity, customer_demand, cost_multiplier=200,
                                    constraint_1_multiplier=constraint_1_multiplier, constraint_2_multiplier=constraint_2_multiplier)
    
    qubo_len, q_energy, q_clusters = qubo_clustering.fit(num_reads=num_reads)
    
    silhouette_score_qubo = silhouette_score(customer_locations, q_clusters)
    
    
    q_unassigned_nodes = q_clusters.count(-1)
    q_demand_errors = count_clusters_with_more_demand(q_clusters, len(vehicle_capacity), vehicle_capacity[0], customer_demand)
    print(f'Silhouette Score: {silhouette_score_qubo}')
    print(f'Unassigned nodes: {q_unassigned_nodes}')
    print(f'Demand errors: {q_demand_errors}')
    
    plot_data(q_clusters, len(vehicle_capacity), vehicle_capacity[0], customer_locations, depot,
        customer_demand, show_demand=False, show_cluster_demand=True, save_file=save_file)

    
    return q_clusters
    
def run_tsp_solver(cluster, customer_locations):
    cluster_distances = compute_tsp_cluster_distances(customer_locations, cluster)
    solution = solve_tsp(cluster_distances, 0)
    parsed_solution = []
    for pair in solution[0]:
        if pair[0]==0:
            if pair[1]==0:
                new_pair = (0,0)
            else:
                new_pair = (0, cluster[pair[1]-1]+1)
        else:
            if pair[1]==0:
                new_pair = (cluster[pair[0]-1]+1, 0)
            else:
                new_pair = (cluster[pair[0]-1]+1, cluster[pair[1]-1]+1)
        parsed_solution += [new_pair]
        
    return parsed_solution

def run_classic_solvers(clusters, customer_locations):
    # Run TSP OR-Tools solvers for each cluster
    solutions = []
    pool = mp.Pool(mp.cpu_count())
    result_objects = [pool.apply_async(run_tsp_solver, args=(cluster, customer_locations[:-1])) for cluster in clusters]
    for r in result_objects:
        solutions += [r.get()]
    pool.close()
    pool.join()
    
    return solutions
    
def main():
    # Do stuff
    problems = ['CMT01.xml','CMT02.xml','CMT03.xml','CMT04.xml','CMT05.xml']
    for problem in problems:
        problem_name = problem.split(".xml")[0]
        customer_demand, vehicle_capacity, distances, customer_locations, depot = read_xml(
            './data/'+problem,0, False, False)
        print(problem_name)
        
        # QUBO Clustering
        num_reads = 1000
        constraint_1_multiplier = 50000
        constraint_2_multiplier = 20
        
        save_file_qubo = "./results/hybrid_algorithm/"+problem_name+"_clustering.png"
        labels = run_qubo_clustering(distances, vehicle_capacity, customer_demand, customer_locations, depot,
                                       save_file_qubo,  num_reads, constraint_1_multiplier, constraint_2_multiplier)
        
        clusters = [[] for x in range(len(vehicle_capacity))]
        for i,x in enumerate(labels):
          clusters[x] += [i]
        #print(clusters)
        
        customer_demand, vehicle_capacity, distances, customer_locations, depot = read_xml(
            './data/'+problem,0, True, True)
        
        # OR-Tools Routing
        o_solutions = run_classic_solvers(clusters, customer_locations)
        
        if any(o_solutions):
            o_total_distance = calculate_total_distance(o_solutions, distances)
            o_errors = check_solution(o_solutions, 0, 0, customer_demand, vehicle_capacity, verbose = False)
            save_file_or = "./results/hybrid_algorithm/"+problem_name+"_routing.png"
            
            plot_data(labels, len(vehicle_capacity), vehicle_capacity[0], customer_locations, depot,
              customer_demand, show_demand=False, show_cluster_demand=True, solution=o_solutions, save_file=save_file_or)
            print("Total distance: ", o_total_distance)
            print("# Errors: ", o_errors)
        else:
            print("No solution found!")
      
      
if __name__ == "__main__":
    main()