import multiprocessing as mp

from hybrid.clustering.KMedoids import KMedoids
from hybrid.routing.TSP_Solver import TSP_Solver
from hybrid.routing.TSP_Solver_DADK import TSP_Solver_DADK
from solver.Google_OR_Solver import solve_tsp

from helpers.utils import read_xml, get_distances_wo_depots, calculate_total_distance, compute_cluster_distances,get_files_in_folder, process_tsp_solution, compute_tsp_cluster_distances
from helpers.check_solution import check_solution

import time

num_reads_qubo = 10000

def run_solver(cluster, customer_locations):

    num_customers = len(cluster)
    cluster_distances = compute_cluster_distances(customer_locations, cluster)

    solver = TSP_Solver(cluster_distances, cost_multiplier=100, constraint_1_multiplier=100, constraint_2_multiplier=5)

    solution, energy = solver.get_solution(num_reads=num_reads_qubo)

    raw_solution, solution_processed = process_tsp_solution(solution, num_customers, cluster)
    return raw_solution, solution_processed, energy

def run_dadk_solver(cluster, customer_locations):

    num_customers = len(cluster)
    cluster_distances = compute_cluster_distances(customer_locations, cluster)

    solver = TSP_Solver_DADK(cluster_distances, constraint_1_multiplier=100, constraint_2_multiplier=600)

    solution, energy = solver.get_solution(num_reads=num_reads_qubo)

    raw_solution, solution_processed = process_tsp_solution(solution, num_customers, cluster)
    
    return raw_solution, solution_processed, energy

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
    
def run_quantum_solvers(clusters, customer_locations):
    # Run TSP QUBO solvers for each cluster
    solutions = []
    raw_solutions = []
    total_energy = 0
    pool = mp.Pool(mp.cpu_count())
    result_objects = [pool.apply_async(run_dadk_solver, args=(cluster, customer_locations)) 
                    for cluster in clusters]
    for i,r in enumerate(result_objects):
        #print('Vehicle',i,'->', r.get()[1])
        raw_solutions += [r.get()[0]]
        solutions += [r.get()[1]]
        total_energy += r.get()[2]
    pool.close()
    pool.join()
    
    return solutions, total_energy


def main():
    # Do stuff
    print("Problem,Nodes,Clusters,Algorithm,Num Iters,Time,Errors,Distance,Energy")
    
    #files = (get_files_in_folder('./data/'))
    #files.sort()
    
    files = ['CMT01.xml','CMT02.xml','CMT03.xml','CMT04.xml','CMT05.xml','CMT06.xml','CMT07.xml','CMT08.xml',
                    'CMT09.xml','CMT10.xml','CMT11.xml','CMT12.xml','CMT13.xml','CMT14.xml',
                    'M-n101-k10.xml','M-n121-k07.xml','M-n151-k12.xml','M-n200-k16.xml','M-n200-k17.xml',
                    'Golden_01.xml','Golden_02.xml','Golden_03.xml','Golden_04.xml','Golden_05.xml','Golden_06.xml',
                    'Golden_07.xml','Golden_08.xml','Golden_09.xml','Golden_10.xml','Golden_11.xml','Golden_12.xml',
                    'Golden_13.xml','Golden_14.xml','Golden_15.xml','Golden_16.xml','Golden_17.xml','Golden_18.xml',
                    'Golden_19.xml','Golden_20.xml',
                    'Li_21.xml','Li_22.xml','Li_23.xml','Li_24.xml','Li_25.xml','Li_26.xml','Li_27.xml','Li_28.xml',
                    'Li_29.xml','Li_30.xml','Li_31.xml','Li_32.xml',
                    'X-n101-k25.xml']
    
    x_files = ['X-n101-k25.xml','X-n106-k14.xml','X-n110-k13.xml','X-n115-k10.xml',
            'X-n120-k6.xml','X-n125-k30.xml','X-n129-k18.xml','X-n134-k13.xml',
            'X-n139-k10.xml','X-n143-k7.xml','X-n148-k46.xml','X-n153-k22.xml',
            'X-n157-k13.xml','X-n162-k11.xml','X-n167-k10.xml','X-n172-k51.xml',
            'X-n176-k26.xml','X-n181-k23.xml','X-n186-k15.xml','X-n190-k8.xml',
            'X-n195-k51.xml','X-n200-k36.xml','X-n204-k19.xml','X-n209-k16.xml',
            'X-n214-k11.xml','X-n219-k73.xml','X-n223-k34.xml','X-n228-k23.xml',
            'X-n233-k16.xml','X-n237-k14.xml','X-n242-k48.xml','X-n247-k47.xml',
            'X-n251-k28.xml','X-n256-k16.xml','X-n261-k13.xml','X-n266-k58.xml',
            'X-n270-k35.xml','X-n275-k28.xml','X-n280-k17.xml','X-n284-k15.xml',
            'X-n289-k60.xml','X-n294-k50.xml','X-n298-k31.xml','X-n303-k21.xml',
            'X-n308-k13.xml','X-n313-k71.xml','X-n317-k53.xml','X-n322-k28.xml',
            'X-n327-k20.xml','X-n331-k15.xml','X-n336-k84.xml','X-n344-k43.xml',
            'X-n351-k40.xml','X-n359-k29.xml','X-n367-k17.xml','X-n376-k94.xml',
            'X-n384-k52.xml','X-n393-k38.xml','X-n401-k29.xml','X-n411-k19.xml',
            'X-n420-k130.xml','X-n429-k61.xml','X-n439-k37.xml','X-n449-k29.xml',
            'X-n459-k26.xml','X-n469-k138.xml','X-n480-k70.xml','X-n491-k59.xml',
            'X-n502-k39.xml','X-n513-k21.xml','X-n524-k137.xml','X-n536-k96.xml',
            'X-n548-k50.xml','X-n561-k42.xml','X-n573-k30.xml','X-n586-k159.xml',
            'X-n599-k92.xml','X-n613-k62.xml','X-n627-k43.xml','X-n641-k35.xml',
            'X-n655-k131.xml','X-n670-k126.xml','X-n685-k75.xml','X-n701-k44.xml',
            'X-n716-k35.xml','X-n733-k159.xml','X-n749-k98.xml','X-n766-k71.xml',
            'X-n783-k48.xml','X-n801-k40.xml','X-n819-k171.xml','X-n837-k142.xml',
            'X-n856-k95.xml','X-n876-k59.xml','X-n895-k37.xml','X-n916-k207.xml',
            'X-n936-k151.xml','X-n957-k87.xml','X-n979-k58.xml','X-n1001-k43.xml']
    
    for problem in files[:1]:
        problem_name = problem.split(".xml")[0]
        customer_demand, vehicle_capacity, distances, customer_locations, depot = read_xml(
                './data/'+problem,0, True, True)

        distances_wo_depots = get_distances_wo_depots(distances)
        # KMedoids Algorithm
        if problem == "CMT01.xml":
            demand_penalty = 1
        else:
            demand_penalty = 10000

        iters = 200
        kmedoids = KMedoids(distances_wo_depots, len(vehicle_capacity), customer_demand, vehicle_capacity[0], iters=iters,
                                demand_penalty=demand_penalty, verbose=False)
        clusters, _ = kmedoids.fit()
        
        print(clusters)
    
        # QUBO Routing
        #q_start = time.time()
        #q_solutions, total_energy = run_quantum_solvers(clusters, customer_locations)
        #q_end = time.time()
        #q_total_distance = calculate_total_distance(q_solutions, distances)
        #q_errors = check_solution(q_solutions, 0, 0, customer_demand, vehicle_capacity, verbose = False)
        #save_file_qubo = "./results/qubo_routing/"+problem_name+"_qubo_routing.png"
        #kmedoids.plot_data(clusters, customer_locations[1:], depot, q_solutions, save_file=save_file_qubo)
        #print("{},{},{},{},{},{:.2f},{},{:.20f},{:.5f}".format(problem,len(customer_demand), len(vehicle_capacity), "QUBO", num_reads_qubo,
        #                                                q_end-q_start, q_errors, q_total_distance, total_energy))
        
        # OR-Tools Routing
        o_start = time.time()
        o_solutions = run_classic_solvers(clusters, customer_locations)
        o_end = time.time()
        if any(o_solutions):
            o_total_distance = calculate_total_distance(o_solutions, distances)
            o_errors = check_solution(o_solutions, 0, 0, customer_demand, vehicle_capacity, verbose = False)
            if problem_name[0]=="X":
                save_file_or = "./results/or_routing/X_datasets/"+problem_name+"_or_routing.png"
            else:
                save_file_or = "./results/or_routing/"+problem_name+"_or_routing.png"
            kmedoids.plot_data(clusters, customer_locations[1:], depot, o_solutions, save_file=save_file_or)
            print("{},{},{},{},{},{:.2f},{},{:.20f},{:.5f}".format(problem,len(customer_demand), len(vehicle_capacity), "OR-Tools", 0,
                                                            o_end-o_start, o_errors, o_total_distance, 0))
        else:
            print("{},{},{},{},{},{:.2f},{},{:.20f},{:.5f}".format(problem,len(customer_demand), len(vehicle_capacity), "OR-Tools", 0,
                                                            o_end-o_start, -1, -1, 0))
      
if __name__ == "__main__":
    main()