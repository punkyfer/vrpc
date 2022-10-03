from pyqubo import Array, Constraint, Placeholder, solve_qubo

import neal

class TSP_Solver:
    
    def __init__(self, distance_matrix, cost_multiplier=1, constraint_1_multiplier=1, constraint_2_multiplier=1):
        self.num_customers = len(distance_matrix)-2
        self.all_vertices = self.num_customers+1
        self.distance_matrix = distance_matrix
        self.cost_multiplier = cost_multiplier
        self.constraint_1_multiplier = constraint_1_multiplier
        self.constraint_2_multiplier = constraint_2_multiplier
        
    def build_qubo_model(self):
    
        # Create the set of variables
        vrp_arr = Array.create("vrp_arr", shape=(self.all_vertices, self.all_vertices), vartype="BINARY")
        
        
        # Hamiltonian Cycles

        # Every vertex can only appear once in a cycle
        ct1 = 0
        for v in range(self.all_vertices):
            ct1_t = 1
            for j in range(self.all_vertices):
                ct1_t += -1*vrp_arr[j][v]
            ct1_t = ct1_t**2
            ct1 += Constraint(ct1_t, label="Node {} must only appear once in the cycle".format(v))
            
        # There must be a jth node in the cycle for each j
        ct2 = 0
        for j in range(self.all_vertices):
            ct2_t = 1
            for v in range(self.all_vertices):
                ct2_t += -1*vrp_arr[j][v]
            ct2_t = ct2_t**2
            ct2 += Constraint(ct2_t, label="There must be a {}th node in the cycle".format(v))
            
        # The cycle must start at vertex 0
        ct3_t = 1
        ct3_t += -1*vrp_arr[0][0]
        ct3_t = ct3_t ** 2
        ct3 = Constraint(ct3_t, label="The cycle must start at vertex 0")
        
        H_A = ct1*self.constraint_1_multiplier + ct2*self.constraint_1_multiplier + ct3*self.constraint_1_multiplier
        
        
        # TSP 
        H_B = 0
        for u in range(self.all_vertices):
            for v in range(self.all_vertices):
                for j in range(self.all_vertices-1):
                    if u!=v:
                        H_B += self.distance_matrix[u][v]*self.cost_multiplier * vrp_arr[j][u]*vrp_arr[j+1][v]
            
        final_eq = H_A*self.constraint_1_multiplier + H_B*self.constraint_2_multiplier
        
        model = final_eq.compile()
    
        qubo, offset = model.to_qubo()
        
        return qubo, model
    
    def get_solution(self, num_reads=10000, verbose=False):
        qubo, model = self.build_qubo_model()
        sampler = neal.SimulatedAnnealingSampler()
        sa_solution = sampler.sample_qubo(qubo, num_reads=num_reads)
        solution = sa_solution.first
        energy = solution.energy
            
        return solution.sample.values(), energy