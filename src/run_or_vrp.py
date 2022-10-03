
from ast import parse
from solver.Google_OR_Solver import solve_cvrp, solve_tsp
import time
from helpers.utils import read_xml, calculate_total_distance, get_files_in_folder, get_distances_wo_depots, compute_tsp_cluster_distances, plot_data, get_cluster_list
from helpers.check_solution import check_solution
import time

import sys

from hybrid.clustering.KMedoids import KMedoids

import multiprocessing as mp

def run_tsp_solver(cluster, customer_locations):
    cluster_distances = compute_tsp_cluster_distances(customer_locations, cluster)
    solution = solve_tsp(cluster_distances, 0)
    parsed_solution = []
    #print(solution)
    #print(cluster)
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

def main():
    # Do stuff
    
    cluster = True
    if len(sys.argv)>1 and sys.argv[1]=='cluster':
            cluster = True
    
    files = (get_files_in_folder('./data/'))
    files.sort()
    print("Problem,Customers,Vehicles,Time,Errors,Distance")
    #for problem in ['Juan_10.xml',
    #                'CMT01.xml','CMT02.xml','CMT03.xml','CMT04.xml','CMT05.xml','CMT06.xml','CMT07.xml','CMT08.xml',
    #                'CMT09.xml','CMT10.xml','CMT11.xml','CMT12.xml','CMT13.xml','CMT14.xml',
    #                'M-n101-k10.xml','M-n121-k07.xml','M-n151-k12.xml','M-n200-k16.xml','M-n200-k17.xml',
    #                'Golden_01.xml','Golden_02.xml','Golden_03.xml','Golden_04.xml','Golden_05.xml','Golden_06.xml',
    #                'Golden_07.xml','Golden_08.xml','Golden_09.xml','Golden_10.xml','Golden_11.xml','Golden_12.xml',
    #                'Golden_13.xml','Golden_14.xml','Golden_15.xml','Golden_16.xml','Golden_17.xml','Golden_18.xml',
    #                'Golden_19.xml','Golden_20.xml',
    #                'Li_21.xml','Li_22.xml','Li_23.xml','Li_24.xml','Li_25.xml','Li_26.xml','Li_27.xml','Li_28.xml',
    #                'Li_29.xml','Li_30.xml','Li_31.xml','Li_32.xml',
    #                'X-n101-k25.xml',
    #                'VeRoLogV08_16.xml','VeRoLogV12_15.xml']:
    files = ['CMT01.xml']
    for problem in files:
        # Read data
        if cluster:
            customer_demand, vehicle_capacity, distances, customer_locations, depot = read_xml(
                './data/'+problem,0, True, True)
        else:
            customer_demand, vehicle_capacity, distances, customer_locations, depot = read_xml(
                './data/'+problem,0, True, False)
        
        if problem == "CMT01.xml" or problem == "CMT05.xml":
            demand_penalty = 1
        else:
            demand_penalty = 10000
            
        distances_wo_depots = get_distances_wo_depots(distances)
        
        print(customer_demand)
        
        
        start = time.time()
        if cluster:
            kmedoids = KMedoids(distances_wo_depots, len(vehicle_capacity), customer_demand, vehicle_capacity[0], 200, 
                                demand_penalty=demand_penalty, verbose=False)
            clusters = kmedoids.fit()
            #solutions = []
            #for cluster in clusters:
            #    solution = run_tsp_solver(cluster, customer_locations[:-1])
            #    solutions += [solution]
            solutions = []
            pool = mp.Pool(mp.cpu_count())
            result_objects = [pool.apply_async(run_tsp_solver, args=(cluster, customer_locations[:-1])) for cluster in clusters]
            for r in result_objects:
                solutions += [r.get()]
            pool.close()
            pool.join()
        else:
            # Run VRPC solver
            solutions = solve_cvrp([0]+customer_demand, vehicle_capacity, distances, depot)
        end = time.time()
        if solutions!=None:
            #print("----------")
            print(solutions)
            print("Total distance of all routes:", calculate_total_distance(solutions, distances))
            total_distance = calculate_total_distance(solutions, distances)
            errors = check_solution(solutions, 0, 0, customer_demand, vehicle_capacity, check_capacity=False, verbose = False)
            print("{},{},{},{:.2f},{},{:.2f}".format(problem,len(customer_demand), len(vehicle_capacity), 
                                                        end-start, errors, total_distance))
            num_customers = max(max(clusters))+1
            plot_data(get_cluster_list(clusters, num_customers), len(vehicle_capacity), vehicle_capacity[0], customer_locations[1:-1], depot, 
                customer_demand, solutions, show_demand=False, show_cluster_demand=False)
        else:
            print("{},{},{},{:.2f},No Solution".format(problem,len(customer_demand), len(vehicle_capacity), 
                                                            end-start))
    
if __name__ == "__main__":
    main()