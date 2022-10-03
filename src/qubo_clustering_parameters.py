from hybrid.clustering.QUBO_Clustering import QUBO_Clustering

from helpers.utils import read_xml, plot_data, count_clusters_with_more_demand

from sklearn.metrics import silhouette_score

import time

import multiprocessing as mp

import csv

def run_qubo_solver(distances, vehicle_capacity, customer_demand, num_reads, constraint_1_multiplier, constraint_2_multiplier):
  qstart = time.time()
  
  qubo_clustering = QUBO_Clustering(distances, vehicle_capacity, customer_demand, cost_multiplier=200,
                                  constraint_1_multiplier=constraint_1_multiplier, constraint_2_multiplier=constraint_2_multiplier)
  
  qubo_len, q_energy, q_clusters = qubo_clustering.fit(num_reads=num_reads)
  
  qend = time.time()
  
  return num_reads, qubo_len, q_energy, q_clusters, qend-qstart

def run_qubo_clustering(problem):
    problem_name = problem.split(".xml")[0]

    csv_file_name = problem_name+'_Parameters.csv'    
    f = open('./results/qubo_clustering_num_reads/'+csv_file_name, 'w')
    writer = csv.writer(f)
    
    headers = "Problem,Nodes,Clusters,Algorithm,Solver,Num Reads/Iters,Time,Unassigned Nodes,Demand Errors,Silhouette,QUBO length,Energy,Constraint 1,Constraint 2".split(",")
    writer.writerow(headers)
    
    customer_demand, vehicle_capacity, distances, customer_locations, depot = read_xml(
            './data/'+problem,0, False, False)
    
    num_reads = 1000
    constraint_1_multiplier = 50000
    constraint_2_multiplier = 20
      
    pool = mp.Pool(mp.cpu_count())
    result_objects = [pool.apply_async(run_qubo_solver, args=(distances, vehicle_capacity, customer_demand, num_reads, constraint_1_multiplier, constraint_2_multiplier)) 
                    for num_reads in [10,100,500,1000,2000,5000,10000]]
    for i,r in enumerate(result_objects):
        num_reads = r.get()[0]
        qubo_len = r.get()[1]
        q_energy = r.get()[2]
        q_clusters = r.get()[3]
        q_time = r.get()[4]
        save_file = "./results/qubo_clustering_num_reads/{}/{}_nr({}).png".format(problem_name,problem_name, num_reads)
        silhouette_score_qubo = silhouette_score(customer_locations, q_clusters)
        
        q_unassigned_nodes = q_clusters.count(-1)
        q_demand_errors = count_clusters_with_more_demand(q_clusters, len(vehicle_capacity), vehicle_capacity[0], customer_demand)
        
        plot_data(q_clusters, len(vehicle_capacity), vehicle_capacity[0], customer_locations, depot,
            customer_demand, show_demand=False, show_cluster_demand=True, save_file=save_file)
        
        writer.writerow("{},{},{},{},{},{},{:.2f},{},{},{:.20f},{},{},{},{}".format(problem,len(customer_demand), len(vehicle_capacity), "QUBO", "SimulatedAnnealingSampler",
                                                                num_reads,q_time, q_unassigned_nodes, q_demand_errors, silhouette_score_qubo, qubo_len,
                                                                q_energy, constraint_1_multiplier, constraint_2_multiplier).split(","))
    pool.close()
    pool.join()
    f.close()
      
      

def main():
    # Do stuff
    #files = (get_files_in_folder('./data/'))
    #files.sort()
    #files =['CMT01.xml','M-n101-k10.xml','M-n121-k07.xml','M-n151-k12.xml','M-n200-k16.xml','M-n200-k17.xml']
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
    
    X_files = ['X-n101-k25.xml','X-n106-k14.xml','X-n110-k13.xml','X-n115-k10.xml',
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
    
    problems = ['CMT06.xml','CMT03.xml','CMT01.xml','CMT02.xml','CMT11.xml','CMT12.xml','M-n101-k10.xml','M-n121-k07.xml','X-n106-k14.xml','X-n110-k13.xml','X-n120-k6.xml']
    for problem in problems[:1]:
      print(problem)
      run_qubo_clustering(problem)
      print("done")
      
if __name__ == "__main__":
    main()