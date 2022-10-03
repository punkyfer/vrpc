from pyqubo import Array, Constraint, Placeholder, solve_qubo
import neal
from dwave_qbsolv import QBSolv
import math

class QUBO_Clustering:
  def __init__(self, distances, vehicle_capacity, customer_demand,
                cost_multiplier, constraint_1_multiplier,
                constraint_2_multiplier):
    self.num_customers = len(customer_demand)
    self.num_clusters = len(vehicle_capacity)
    self.distances = distances
    self.vehicle_capacity = vehicle_capacity
    self.customer_demand = customer_demand
    self.cost_multiplier = cost_multiplier
    self.constraint_1_multiplier = constraint_1_multiplier
    self.constraint_2_multiplier = constraint_2_multiplier
    self.min_demand = min(customer_demand)
        
  def build_qubo_model(self):
    
    x = Array.create("x", shape=(self.num_customers, self.num_clusters), vartype="BINARY")
    slack = Array.create("slack", shape=(self.num_clusters, math.ceil((self.vehicle_capacity[0]-1)/self.min_demand)), vartype="BINARY")
    
    # Main Equation
    M = 0
    for k in range(self.num_clusters):
        for i in range(self.num_customers-1):
            for j in range(i+1,self.num_customers):
                M += self.distances[i][j] * self.cost_multiplier * x[i][k] * x[j][k]
    
    # Each vertex must be assigned to one cluster
    C1 = 0
    for i in range(self.num_customers):
        C1_t = -1
        for k in range(self.num_clusters):
            C1_t += x[i][k]
        C1_t = C1_t**2
        C1 += Constraint(C1_t, label="Customer {} not assigned to any cluster".format(i))
        
    # Sum of demand in each cluster must not exceed vehicle capacity
    C2 = 0
    for k in range(self.num_clusters):
        C2_t = -self.vehicle_capacity[0]
        for i in range(self.num_customers):
            C2_t += self.customer_demand[i]*x[i][k]
        for l, v in enumerate(range(math.ceil((self.vehicle_capacity[0]-1)/self.min_demand))):
            C2_t += (l+1)*self.min_demand*slack[k][v]
        C2_t = C2_t**2
        C2 += Constraint(C2_t, label="Cluster {} exceeds vehicle capacity {}".format(k, self.vehicle_capacity[k]))
        
                
    H = M + C1 * self.constraint_1_multiplier + C2 * self.constraint_2_multiplier
    model = H.compile()
    
    qubo, offset = model.to_qubo()
    
    return qubo, model
  
  def fit(self, num_reads=1000, verbose=False):
    
    qubo_len = self.num_customers*self.num_clusters + self.num_clusters * math.ceil((self.vehicle_capacity[0]-1)/self.min_demand)
    
    qubo, model = self.build_qubo_model()
    sampler = neal.SimulatedAnnealingSampler()
    sa_solution = sampler.sample_qubo(qubo, num_reads=num_reads)
    agg_solution = sa_solution.aggregate()
    raw_solution = agg_solution.first.sample
    energy = agg_solution.first.energy
    # decode for easier analysis
    decoded_samples = model.decode_sample(raw_solution, vartype="BINARY")
    if verbose:
      print("Energy:",energy)
      # Show failed constraints
      print(decoded_samples.constraints(only_broken=True))
    # extract label
    customer_clusters = [-1]*self.num_customers
    for k in range(self.num_clusters):
        for i in range(self.num_customers):
            if decoded_samples.array("x", (i,k)) == 1:
                customer_clusters[i] = k
    return qubo_len,energy,customer_clusters