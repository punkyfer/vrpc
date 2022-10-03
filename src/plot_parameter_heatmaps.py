from helpers.utils import plot_routing_parameter_heatmaps, plot_clustering_parameter_heatmaps

def main():
    # Do stuff
    #files = (get_files_in_folder('./data/'))
    #files.sort()
    #files =['CMT01.xml','M-n101-k10.xml','M-n121-k07.xml','M-n151-k12.xml','M-n200-k16.xml','M-n200-k17.xml']
    files = ['CMT01_Parameters.csv','CMT02_Parameters.csv','CMT03_Parameters.csv','CMT04_Parameters.csv','CMT06_Parameters.csv',
                    'CMT11_Parameters.csv','CMT12_Parameters.csv',
                    'M-n101-k10_Parameters.csv','M-n121-k07_Parameters.csv',
                    'X-n106-k14_Parameters.csv','X-n110-k13_Parameters.csv']
    
    csv_files = []
    for problem in files:
      problem_name = problem.split("_")[0]
      csv_file = "./results/qubo_routing_parameters/"+problem
      csv_files += [csv_file]
      save_file = "./results/qubo_routing_parameters/heatmaps/"+problem_name+"_rt_multiplier_heatmap.png"
      plot_routing_parameter_heatmaps(problem_name, csv_file, save_file)
      print("done")
      
    save_file = "./results/qubo_routing_parameters/heatmaps/Average_rt_multiplier_heatmap.png"
    plot_routing_parameter_heatmaps("Average", csv_files, save_file)
    
      
if __name__ == "__main__":
    main()