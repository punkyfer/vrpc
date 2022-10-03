import neal

from dwave.system import DWaveSampler, EmbeddingComposite

from dadk.BinPol import BitArrayShape, VarSlack, VarShapeSet, SlackType, BinPol

from helpers.utils import powerset

class QUBO_Solver_DADK:
  def __init__(self, distances, vehicle_capacity, customer_demand,
               cost_multiplier, constraint_1_multiplier, constraint_2_multiplier, 
               constraint_3_multiplier, constraint_4_multiplier,
               constraint_5_multiplier, constraint_6_multiplier):
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
    
  def qubo_len(self):
      vars_powerset = 0
      for s in self.powerset:
          vars_powerset += len(s)-1
          
      vars_capacity = 0
      for i in range(self.num_vehicles):
          vars_capacity += self.vehicle_capacity[i]
          
      print("Decision Variables: ",(self.all_vertices * self.all_vertices * self.num_vehicles))
      print("Capacity slack: ", vars_capacity)
      print("CLS slack: ", vars_powerset)
      return (self.all_vertices * self.all_vertices * self.num_vehicles) + vars_capacity + vars_powerset

  def build_qubo(self):
    
    # Create the set of variables
    my_bit_shape_array = BitArrayShape(name='vrp_arr', shape=(self.all_vertices, self.all_vertices, self.num_vehicles), 
                                       axis_names=['customer1','customer2', 'vehicle']) 
    # Create slack variables
    slack_list = []
    # subtour elimination
    for i, s in enumerate(self.powerset):
        slack_list.append(VarSlack(name='slack_variable_c_6_'+str(i),start=0,step=1,
                               stop=len(s)-1, slack_type=SlackType.binary))
    # vehicle capacity
    for i in range(self.num_vehicles):
        slack_list.append(VarSlack(name='slack_variable_c_5_'+str(i),start=0,step=1,
                                stop=self.vehicle_capacity[i], slack_type=SlackType.binary))
    
    my_varshapeset = VarShapeSet(my_bit_shape_array, *slack_list)
    
    
    # Main Equation

    eq = BinPol(my_varshapeset)

    for k in range(self.num_vehicles):
        for i in range(self.all_vertices):
            for j in range(self.all_vertices):
                eq.add_term(self.distances[i][j]*self.cost_multiplier,("vrp_arr",i,j,k))

    # Constraints

    # Each customer is visited only once
    ct1 = BinPol(my_varshapeset)
    for i in range(1,self.num_customers+1):
        ct1_t = BinPol(my_varshapeset)
        ct1_t.add_term(1)
        for k in range(self.num_vehicles):
            for j in range(self.all_vertices):
                if(i!=j):
                    ct1_t.add_term(-1,("vrp_arr",i,j,k))
        ct1_t = ct1_t**2
        ct1 += ct1_t

    # Each vehicle must start at depot 0
    ct2 = BinPol(my_varshapeset)
    for k in range(self.num_vehicles):
        ct2_t = BinPol(my_varshapeset)
        ct2_t.add_term(1)
        for j in range(self.all_vertices):
            ct2_t.add_term(-1, ("vrp_arr",0,j,k))
        ct2_t = ct2_t**2
        ct2 += ct2_t

    # Each vehicle must end at depot n+1
    ct3 = BinPol(my_varshapeset)
    for k in range(self.num_vehicles):
        ct3_t = BinPol(my_varshapeset)
        ct3_t.add_term(1)
        for i in range(self.all_vertices):
            ct3_t.add_term(-1, ("vrp_arr",i,self.num_customers+1,k))
        ct3_t = ct3_t**2
        ct3 += ct3_t

    # After a vehicle arrives at i it must leave i for j
    ct4 = BinPol(my_varshapeset)
    for h in range(1,self.num_customers+1):
        for k in range(self.num_vehicles):
            ct4_t = BinPol(my_varshapeset)
            for i in range(self.all_vertices):
                if (i!=h):
                    ct4_t.add_term(1, ("vrp_arr",i,h,k))
            for j in range(self.all_vertices):
                if (j!=h):
                    ct4_t.add_term(-1, ("vrp_arr",h,j,k))
            ct4_t = ct4_t**2
            ct4 += ct4_t

    # A vehicle can only carry up to its capacity
    ct5 = BinPol(my_varshapeset)
    for k in range(self.num_vehicles):
        ct5_t = BinPol(my_varshapeset)
        for i in range(1, self.num_customers+1):
            for j in range(1, self.num_customers+1):
                if(i!=j):
                    ct5_t.add_term(self.customer_demand[i-1], ("vrp_arr",i,j,k))
        ct5 += ((ct5_t-self.vehicle_capacity[k]).add_slack_variable('slack_variable_c_5_'+str(k),factor=1))**2

    # Closed loop subtour elimination
    ct6 = BinPol(my_varshapeset)
    for e,s in enumerate(self.powerset):
        ct6_t = BinPol(my_varshapeset)
        for k in range(self.num_vehicles):
            for i in range(len(s)):
                for j in range(i+1,len(s)):
                    ct6_t.add_term(1, ("vrp_arr",i,j,k))
        ct6 += ((ct6_t-(len(self.powerset)-1)).add_slack_variable('slack_variable_c_6_'+str(e),factor=1))**2
        
        
    final_eq = BinPol.sum(eq + ct1*self.constraint_1_multiplier,
                          ct2*self.constraint_2_multiplier,
                          ct3*self.constraint_3_multiplier,
                          ct4*self.constraint_4_multiplier,
                          ct5*self.constraint_5_multiplier,
                          ct6*self.constraint_6_multiplier,
                         )

    return final_eq
  
  def get_bqm(self):
        return self.build_qubo().as_bqm()
      
  def get_solution(self, num_reads=10000):
        bqm = self.get_bqm()
    
        sa = neal.SimulatedAnnealingSampler()
        
        sampleset = sa.sample(bqm, num_reads=num_reads)
        solution = sampleset.aggregate().first
        
        solution_processed = [[] for _ in range(self.num_vehicles)]
        tuples = []
        for i in range(self.num_customers+2):
            for j in range(self.num_customers+2):
                for k in range(self.num_vehicles):
                    tuples += [(i,j,k)]

        for i, val in enumerate(solution.sample.values()):
            if val==1 and i < len(tuples):
                solution_processed[tuples[i][2]] += [(tuples[i][0],tuples[i][1])]
        
        return solution.energy, solution_processed