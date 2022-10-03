from hybrid.routing.TSP_Solver_DADK import TSP_Solver_DADK
from hybrid.clustering.KMedoids import KMedoids

from helpers.utils import read_xml, compute_cluster_distances, process_tsp_solution, get_distances_wo_depots, calculate_total_distance
from helpers.check_solution import check_solution

from sklearn.metrics import silhouette_score

import time

import multiprocessing as mp

import csv

def run_dadk_solver(cluster, customer_locations, constraint_1_multiplier, constraint_2_multiplier, num_reads):

    num_customers = len(cluster)
    cluster_distances = compute_cluster_distances(customer_locations, cluster)

    solver = TSP_Solver_DADK(cluster_distances, constraint_1_multiplier=constraint_1_multiplier, constraint_2_multiplier=constraint_2_multiplier)

    solution, energy = solver.get_solution(num_reads=num_reads)

    raw_solution, solution_processed = process_tsp_solution(solution, num_customers, cluster)
    
    return raw_solution, solution_processed, energy
  
def run_quantum_solvers(clusters, customer_locations, constraint_1_multiplier, constraint_2_multiplier, num_reads):
    # Run TSP QUBO solvers for each cluster
    solutions = []
    raw_solutions = []
    total_energy = 0
    pool = mp.Pool(mp.cpu_count())
    result_objects = [pool.apply_async(run_dadk_solver, args=(cluster, customer_locations, constraint_1_multiplier, constraint_2_multiplier, num_reads)) 
                    for cluster in clusters]
    for i,r in enumerate(result_objects):
        raw_solutions += [r.get()[0]]
        solutions += [r.get()[1]]
        total_energy += r.get()[2]
    pool.close()
    pool.join()
    
    return solutions, total_energy

def run_qubo_routing(problem):
    problem_name = problem.split(".xml")[0]

    csv_file_name = problem_name+'_Parameters.csv'    
    f = open('./results/qubo_routing_num_reads/'+csv_file_name, 'w')
    writer = csv.writer(f)
    
    headers = "Problem,Nodes,Vehicles,Algorithm,Solver,Num Reads/Iters,Time,Errors,Distance,Energy,Constraint 1,Constraint 2".split(",")
    writer.writerow(headers)
    
    customer_demand, vehicle_capacity, distances, customer_locations, depot = read_xml(
            './data/'+problem,0, True, True)
    
    # KMedoids Algorithm
    if problem == "CMT01.xml":
        demand_penalty = 1
    else:
        demand_penalty = 10000
        
    iters = 200
    distances_wo_depots = get_distances_wo_depots(distances)
    kmedoids = KMedoids(distances_wo_depots, len(vehicle_capacity), customer_demand, vehicle_capacity[0], iters=iters,
                            demand_penalty=demand_penalty, verbose=False)
    clusters, _ = kmedoids.fit()
    
    num_reads = 10000
    constraint_1_multiplier = 150
    constraint_2_multiplier = 700
      
    for num_reads in [10,100,1000,10000,20000,50000,100000]:
      start = time.time()
      solutions, total_energy = run_quantum_solvers(clusters, customer_locations, constraint_1_multiplier, constraint_2_multiplier, num_reads)
      end = time.time()
      total_distance = calculate_total_distance(solutions, distances)
      errors = check_solution(solutions, 0, 0, customer_demand, vehicle_capacity, check_capacity=False, verbose = False)

      save_file = "./results/qubo_routing_num_reads/{}/{}_nr({}).png".format(problem_name,problem_name, num_reads)
      kmedoids.plot_data(clusters, customer_locations[1:], depot, solutions, save_file=save_file)
      
      writer.writerow("{},{},{},{},{},{},{:.2f},{},{:.5f},{:.5f},{},{}".format(problem,len(customer_demand), len(vehicle_capacity), "QUBO", "SimulatedAnnealingSampler",
                                                              num_reads,end-start, errors, total_distance, total_energy,
                                                              constraint_1_multiplier, constraint_2_multiplier).split(","))
    f.close()
      
      

def main():
    # Do stuff
    
    problems = ['CMT01.xml','CMT02.xml','CMT03.xml','CMT04.xml','CMT06.xml','CMT11.xml','CMT12.xml',
                'M-n101-k10.xml','M-n121-k07.xml',
                'X-n106-k14.xml','X-n110-k13.xml','X-n120-k6.xml','Golden_05.xml','M-n151-k12.xml']
    problems = ['CMT07.xml','CMT08.xml','CMT09.xml','CMT13.xml','CMT14.xml','CMT05.xml','CMT10.xml']
    for problem in problems:
      print(problem)
      run_qubo_routing(problem)
      print("done")
      
if __name__ == "__main__":
    main()