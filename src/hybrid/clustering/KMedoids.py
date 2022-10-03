import numpy as np
import matplotlib.pyplot as plt

import time

from helpers.utils import read_xml,get_distances_wo_depots

class KMedoids:
    def __init__(self,distances,num_clusters,demand, capacity, iters, demand_penalty=1,
                 initial_medoids=None, verbose = False):
        self.distances = np.array(distances)
        self.num_clusters = num_clusters
        self.demand = np.array(demand)
        self.capacity = capacity
        self.iters = iters
        if initial_medoids == None:
            self.medoids = self.compute_initial_medoids()
        else:
            self.medoids = initial_medoids
        self.data = list(range(len(demand)))
        self.verbose = verbose
        self.demand_penalty = demand_penalty
        
        self.cluster_cost_dict = {}
    
    def compute_cluster_load(self, cluster):
        return np.sum(self.demand[cluster])
    
    def compute_initial_medoids(self):
        demand_tuples = [(i,x) for i,x in enumerate(self.demand)]
        demand_tuples.sort(key=lambda x:x[1], reverse=True)
        return [i for i,x in demand_tuples[:self.num_clusters]]
    
    def compute_cluster_costs(self, cluster):
        if str(cluster) in self.cluster_cost_dict:
            cost = self.cluster_cost_dict[str(cluster)]
        else:
            if len(cluster)>0:
                cost = np.sum([self.distances[cluster*len(cluster), np.repeat(cluster, len(cluster))]])
                cluster_demand = np.sum(self.demand[cluster])
                cost += abs((self.capacity) - cluster_demand)*self.demand_penalty
                self.cluster_cost_dict[str(cluster)]=cost
            else:
                cost = 0
        return cost
                
    
    def assign_datapoints(self, medoids, data):
        tmp_clusters = [[] for _ in range(len(medoids))]
        for d in data:
            tmp_clusters[self.distances[d, medoids].argmin()]+=[d]
        cst = 0
        for i, cluster in enumerate(tmp_clusters):
            cst += self.compute_cluster_costs(cluster)
        return tmp_clusters, cst
    
    def plot_data(self, clusters, data, depot, paths=None, show_demand=False, plot_numbers=False, save_file=None):
        colors =  np.array(np.random.randint(0, 255, size =(self.num_clusters, 4)))/255
        colors[:,3]=1
        markers = ["o","^","s","p","P","h","X","8","D"]
        
        fig, ax = plt.subplots(1,1)
        total_demand = 0
        for i, x in enumerate(self.medoids):
            cluster_demand = self.compute_cluster_load(clusters[i])
            total_demand += cluster_demand
            if cluster_demand>self.capacity:
                marker = "X"
            else:
                marker = "o"
            #markers[i%len(markers)]
            [plt.scatter(data[t][0], data[t][1], marker=marker, 
                         s=50, color=colors[i]) for t in clusters[i] if t!=x]
            plt.scatter(data[x][0], data[x][1], marker="*", s=100, color="black")
            ax.annotate(cluster_demand, (data[x][0], data[x][1]))
            if paths is not None:
                for path in paths[i]:
                    if path[0]==0:
                        A = depot
                    else:
                        A = data[path[0]-1]
                    if path[1]==0:
                        B = depot
                    else:
                        B = data[path[1]-1]
                    plt.arrow(A[0], A[1], B[0]-A[0],B[1]-A[1], head_width=1, 
                              length_includes_head=True,color=colors[i])
        
        plt.scatter(depot[0], depot[1], marker="o", s=50, color='black')
        
        if show_demand:
            for i, txt in enumerate(self.demand):
                ax.annotate(txt, (data[i][0], data[i][1]))
                
        if plot_numbers:
            for i in range(len(self.data)):
                ax.annotate(i, (data[i][0], data[i][1]))
                
        if save_file:
            if not save_file.endswith(".png"):
                save_file += ".png"
            plt.savefig(save_file)
        else:
            plt.show()
            
        plt.close()
    
    def fit(self):
        count = 0
        clusters, cost = self.assign_datapoints(self.medoids, self.data)
        
        run = True
        
        # TODO: ADD Parallelization
        while run:
            swap = False
            for i,m in enumerate(self.medoids):
                for d in self.data:
                    if d not in self.medoids:
                        tmp_medoids =  self.medoids.copy()
                        tmp_medoids[i] = d
                        
                        tmp_clusters, tmp_cost = self.assign_datapoints(tmp_medoids, self.data)
                        
                        if tmp_cost < cost:
                            swap = True
                            cost = tmp_cost
                            clusters = tmp_clusters
                            self.medoids = tmp_medoids.copy()
            count+=1
            if count>=self.iters:
                if self.verbose:
                    print("End of the iterations.")
                run = False
            if not swap:
                if self.verbose:
                    print("No changes.")
                run = False
        return clusters, count
    
    
def main():
    # Do stuff
    customer_demand, vehicle_capacity, distances, customer_locations, depot = read_xml(
        './data/Golden_03.xml',0, True, True)

    distances_wo_depots = get_distances_wo_depots(distances)
    start = time.time()
    kmedoids = KMedoids(distances_wo_depots, len(vehicle_capacity), customer_demand, vehicle_capacity[0], 200, 
                               demand_penalty=10000, verbose=True)
    clusters = kmedoids.fit()
    end = time.time()
    print("Time elapsed: ", end-start)
    #print(clusters)
    kmedoids.plot_data(clusters, customer_locations[1:-1], depot)
    
if __name__ == "__main__":
    main()