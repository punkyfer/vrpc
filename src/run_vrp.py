import multiprocessing as mp

from helpers.utils import read_xml, process_tsp_solution, get_distances_wo_depots, compute_cluster_distances, calculate_total_distance
from helpers.check_solution import check_solution

from hybrid.clustering.KMedoids import KMedoids
from hybrid.routing.TSP_Solver_DADK import TSP_Solver_DADK

import time

import sys


def run_solver(cluster, v, customer_locations, problem, qpu):

    num_customers = len(cluster)
    cluster_distances = compute_cluster_distances(customer_locations, cluster)

    solver = TSP_Solver_DADK(cluster_distances, cost_multiplier=2, constraint_1_multiplier=100, constraint_2_multiplier=3)

    solution, energy = solver.get_solution(num_reads=10000, label="{} - {}".format(problem, v), qpu=qpu)
    
    print(solution, energy)

    raw_solution, solution_processed = process_tsp_solution(solution, num_customers, cluster)
    return raw_solution, solution_processed, energy

def main():
    # Do stuff
    print("Problem,Customers,Vehicles,Time,Errors,Distance,Energy")
    for problem in ['CMT01.xml','M-n101-k10.xml','M-n121-k07.xml','M-n151-k12.xml','M-n200-k16.xml','M-n200-k17.xml'][:1]:
        start = time.time()
        # Read data
        customer_demand, vehicle_capacity, distances, customer_locations, depot = read_xml(
            './data/'+problem,0, True, True)

        # Create clusters
        if problem == "CMT01.xml":
            demand_penalty = 1
        else:
            demand_penalty = 10000
        distances_wo_depots = get_distances_wo_depots(distances)
        kmedoids = KMedoids(distances_wo_depots, len(vehicle_capacity), customer_demand, vehicle_capacity[0], 200, 
                                demand_penalty=demand_penalty, verbose=False)
        clusters = kmedoids.fit()
        #kmedoids.plot_data(clusters, customer_locations[1:-1], depot)
        
        qpu = False
        if len(sys.argv)>1 and sys.argv[1]=='qpu':
            print("Using QPU")
            qpu = True
        # Run TSP solvers for each cluster
        solutions = []
        raw_solutions = []
        total_energy = 0
        pool = mp.Pool(mp.cpu_count())
        result_objects = [pool.apply_async(run_solver, args=(cluster, v, customer_locations, problem, qpu)) 
                        for v,cluster in enumerate(clusters)]
        for i,r in enumerate(result_objects):
            #print('Vehicle',i,'->', r.get()[1])
            raw_solutions += [r.get()[0]]
            solutions += [r.get()[1]]
            total_energy += r.get()[2]
        end = time.time()
        pool.close()
        pool.join()
        #print(raw_solutions)
        #print(solutions)
        total_distance = calculate_total_distance(solutions, distances)
        #print("Time elapsed:", end-start)
        #print("Total distance of all routes:", total_distance)
        #print("Total energy:", total_energy)

        #kmedoids.plot_data(clusters, customer_locations[1:], depot, solutions)

        errors = check_solution(solutions, 0, 0, customer_demand, vehicle_capacity, verbose = False)
        #print("Total errors:", errors)
        print("{},{},{},{:.2f},{},{:.2f},{:.2f}".format(problem,len(customer_demand), len(vehicle_capacity), 
                                                        end-start, errors, total_distance, total_energy))
    
if __name__ == "__main__":
    main()