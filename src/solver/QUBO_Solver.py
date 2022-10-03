from pyqubo import Array, Constraint, Placeholder, solve_qubo
import numpy as np
import neal

from dwave.system import DWaveSampler, EmbeddingComposite

from helpers.utils import powerset

class QUBO_Solver:
  def __init__(self, distances, vehicle_capacity, customer_demand,
               cost_multiplier, constraint_1_multiplier, constraint_2_multiplier, 
               constraint_3_multiplier, constraint_4_multiplier,
               constraint_5_multiplier, constraint_6_multiplier,):
    self.num_customers = len(customer_demand)
    self.num_vehicles = len(vehicle_capacity)
    self.distances = distances
    self.vehicle_capacity = vehicle_capacity
    self.customer_demand = customer_demand
    self.powerset = [x for x in powerset([i for i in range(1,self.num_customers+1)]) if len(x)>=2]
    self.all_vertices = self.num_customers + 2
    self.cost_multiplier = cost_multiplier
    self.constraint_1_multiplier = constraint_1_multiplier
    self.constraint_2_multiplier = constraint_2_multiplier
    self.constraint_3_multiplier = constraint_3_multiplier
    self.constraint_4_multiplier = constraint_4_multiplier
    self.constraint_5_multiplier = constraint_5_multiplier
    self.constraint_6_multiplier = constraint_6_multiplier

  def build_qubo(self):
    

    x = Array.create("x", shape=(self.all_vertices, self.all_vertices, self.num_vehicles), vartype="BINARY")
    # Create slack variables
    capacity_slack = Array.create("capacity_slack", shape=(self.num_vehicles, self.vehicle_capacity[0]-1), vartype="BINARY")
    # subtour elimination
    powerset_slack = Array.create("powerset_slack", shape=(len(self.powerset), max(len(s) for s in self.powerset)), vartype="BINARY")
    
    # Main Equation

    eq = 0
    for k in range(self.num_vehicles):
        for i in range(self.all_vertices):
            for j in range(self.all_vertices):
                eq += self.distances[i][j]*self.cost_multiplier*x[i][j][k]

    # Constraints

    # Each customer is visited only once
    ct1 = 0
    for i in range(1,self.num_customers+1):
        ct1_t = 1
        for k in range(self.num_vehicles):
            for j in range(self.all_vertices):
                if(i!=j):
                    ct1_t += -1*x[i][j][k]
        ct1_t = ct1_t**2
        ct1 += Constraint(ct1_t, label="Customer {} is visited more than once".format(i))

    # Each vehicle must start at depot 0
    ct2 = 0
    for k in range(self.num_vehicles):
        ct2_t = 1
        for j in range(self.all_vertices):
            ct2_t += -1*x[0][j][k]
        ct2_t = ct2_t**2
        ct2 += Constraint(ct2_t, label="Vehicle {} must start at depot 0".format(k))

    # Each vehicle must end at depot n+1
    ct3 = 0
    for k in range(self.num_vehicles):
        ct3_t = 1
        for i in range(self.all_vertices):
            ct3_t += -1*x[i][self.num_customers+1][k]
        ct3_t = ct3_t**2
        ct3 += Constraint(ct3_t, label="Vehicle {} must start at depot {}".format(k, self.num_customers+1))

    # After a vehicle arrives at i it must leave i for j
    ct4 = 0
    for h in range(1,self.num_customers+1):
        for k in range(self.num_vehicles):
            ct4_t = 0
            for i in range(self.all_vertices):
                if (i!=h):
                    ct4_t += 1*x[i][h][k]
            for j in range(self.all_vertices):
                if (j!=h):
                    ct4_t += -1*x[h][j][k]
            ct4_t = ct4_t**2
            ct4 += Constraint(ct4_t, label="After vehicle {} arrives at {} it must leave".format(k, h))

    # A vehicle can only carry up to its capacity
    ct5 = 0
    for k in range(self.num_vehicles):
        ct5_t = -self.vehicle_capacity
        for i in range(1, self.num_customers+1):
            for j in range(1, self.num_customers+1):
                if(i!=j):
                    ct5_t += self.customer_demand[i-1]*x[i][j][k]
        for l, v in enumerate(range(self.vehicle_capacity[0]-1)):
            ct5_t += (l+1)*capacity_slack[k][v]
        ct5_t = ct5_t**2
        ct5 += Constraint(ct5_t, label="Vehicle {} must not exceed carrying capacity {}".format(k, self.vehicle_capacity[0]))

    # Closed loop subtour elimination
    ct6 = 0
    for e,s in enumerate(powerset):
        ct6_t = -(len(powerset)-1)
        for k in range(self.num_vehicles):
            for i in range(len(s)):
                for j in range(i+1,len(s)):
                    ct6_t += 1 * x[i][j][k]
        for l,v  in enumerate(s):
          ct6_t += (l+1)*powerset_slack[e][v]
        ct6_t = ct6_t**2
        ct6 += Constraint(ct6_t, label="Subtour found")
        
    final_eq = eq + ct1*self.constraint_1_multiplier + ct2*self.constraint_2_multiplier + \
      ct3*self.constraint_3_multiplier + ct4*self.constraint_4_multiplier + ct5*self.constraint_5_multiplier + ct6*self.constraint_6_multiplier

    return final_eq
      
  def get_solution(self, num_reads=10000, verbose=False):
    
        qubo, model = self.build_qubo_model()
        sampler = neal.SimulatedAnnealingSampler()
        sa_solution = sampler.sample_qubo(qubo, num_reads=num_reads)
        raw_solution = sa_solution.first.sample
        energy = sa_solution.first.energy
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
                #print("Cluster",k,", customer",i,"=",decoded_samples.array("x", (i,k)))
        return customer_clusters