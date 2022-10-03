from helpers.utils import read_xml, process_tsp_solution, get_distances_wo_depots, compute_cluster_distances, calculate_total_distance
from helpers.check_solution import check_solution

from solver.QUBO_Solver_DADK import QUBO_Solver_DADK

import time

import sys

def main():
    # Do stuff
    print("Problem,Customers,Vehicles,Time,Errors,Distance,Energy")
    for problem in ['Juan_10.xml']:
        start = time.time()
        # Read data
        customer_demand, vehicle_capacity, distances, customer_locations, depot = read_xml(
            './data/'+problem,0, True, True)

        
        qubo_solver = QUBO_Solver_DADK(distances, vehicle_capacity, customer_demand, cost_multiplier = 1,
                   constraint_1_multiplier=3000,
                   constraint_2_multiplier=100,
                   constraint_3_multiplier=1,
                   constraint_4_multiplier=100,
                   constraint_5_multiplier=200,
                   constraint_6_multiplier=30)
        
        print(qubo_solver.qubo_len())
        
        energy, solutions = qubo_solver.get_solution(num_reads=10)
        end = time.time()
        #print(raw_solutions)
        print(qubo_solver.qubo_len())
        print(solutions)
        total_distance = calculate_total_distance(solutions, distances)
        #print("Time elapsed:", end-start)
        #print("Total distance of all routes:", total_distance)
        #print("Total energy:", total_energy)

        #kmedoids.plot_data(clusters, customer_locations[1:], depot, solutions)

        errors = check_solution(solutions, 0, 0, customer_demand, vehicle_capacity, verbose = True)
        #print("Total errors:", errors)
        print("{},{},{},{:.2f},{},{:.2f},{:.2f}".format(problem,len(customer_demand), len(vehicle_capacity), 
                                                        end-start, errors, total_distance, energy))
    
if __name__ == "__main__":
    main()