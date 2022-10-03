from dadk.BinPol import BitArrayShape, VarSlack, VarShapeSet, SlackType, BinPol
import neal
from dwave.system import DWaveSampler, EmbeddingComposite

class TSP_Solver_DADK:
    
    def __init__(self, distance_matrix, constraint_1_multiplier=1, constraint_2_multiplier=1):
        self.num_customers = len(distance_matrix)-2
        self.all_vertices = self.num_customers+1
        self.distance_matrix = distance_matrix
        self.constraint_1_multiplier = constraint_1_multiplier
        self.constraint_2_multiplier = constraint_2_multiplier
        
    def build_qubo(self):
        my_bit_shape_array = BitArrayShape(name='vrp_arr', shape=(self.all_vertices, self.all_vertices),
                                       axis_names=['order', 'customer'])
        my_varshapeset = VarShapeSet(my_bit_shape_array)
        
        # Hamiltonian Cycles
        # Every vertex can only appear once in a cycle
        ct1 = BinPol(my_varshapeset)
        for v in range(self.all_vertices):
            ct1_t = BinPol(my_varshapeset)
            ct1_t.add_term(1)
            for j in range(self.all_vertices):
                ct1_t.add_term(-1, ("vrp_arr",j,v))
            ct1_t = ct1_t**2
            ct1 += ct1_t
            
        # There must be a jth node in the cycle for each j
        ct2 = BinPol(my_varshapeset)
        for j in range(self.all_vertices):
            ct2_t = BinPol(my_varshapeset)
            ct2_t.add_term(1)
            for v in range(self.all_vertices):
                ct2_t.add_term(-1, ("vrp_arr",j,v))
            ct2_t = ct2_t**2
            ct2 += ct2_t
            
        # The cycle must start at vertex 0
        ct3 = BinPol(my_varshapeset)
        ct3.add_term(1)
        ct3.add_term(-1, ("vrp_arr", 0, 0))
        ct3 = ct3 ** 2
        
        H_A = BinPol.sum(ct1*self.constraint_1_multiplier,
                        ct2*self.constraint_1_multiplier,
                        ct3*self.constraint_1_multiplier)
        
        # TSP 
        H_B = BinPol(my_varshapeset)
        for u in range(self.all_vertices):
            for v in range(self.all_vertices):
                for j in range(self.all_vertices-1):
                    if u!=v:
                        H_B.add_term(self.distance_matrix[u][v],("vrp_arr",j,u),("vrp_arr",j+1,v))
            
        final_eq = BinPol.sum(H_A*self.constraint_1_multiplier, H_B*self.constraint_2_multiplier)

        return final_eq
    
    def get_bqm(self):
        return self.build_qubo().as_bqm()
    
    def get_solution(self, num_reads=10000, label=None):
        bqm = self.get_bqm()
    
        sa = neal.SimulatedAnnealingSampler()
        
        sampleset = sa.sample(bqm, num_reads=num_reads, label=label)
        solution = sampleset.first
        
        return solution.sample.values(), solution.energy