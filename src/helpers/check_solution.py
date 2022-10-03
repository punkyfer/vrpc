def check_constraint1(solution, num_customers, num_vehicles, verbose):
    if verbose:
        print('\nChecking first constraint:')
    customer_visits = [0]*num_customers
    for k in range(num_vehicles):
        for s in solution[k]:
            for i in range(1, num_customers+1):
                if s[1]==i:
                    customer_visits[i-1] += 1
    errors = 0
    for i, cv in enumerate(customer_visits):
        if cv == 0:
            if verbose:
                print('   {} -> customer {} is never visited'.format('FAIL', i+1))
            errors += 1
        else:
            if cv>1:
                if verbose:
                    print('   {} -> customer {} is visited {} times'.format('FAIL', i+1, cv))
                errors += 1
            else:
                if verbose:
                    print('   {} -> customer {} is visited once'.format('OK', i+1))
    return errors

def check_constraint2(solution, depot, num_vehicles, verbose):
    if verbose:
        print('\nChecking second constraint:')
    errors = 0
    for k in range(num_vehicles):
        starts_at_depot = 0
        start_tuple = None
        for s in solution[k]:
            if s[0]==depot:
                starts_at_depot += 1
        if starts_at_depot:
            if verbose:
                print('   {} -> vehicle {} starts at depot {}'.format('OK', k, depot))
        else:
            if verbose:
                print('   {} -> vehicle {} does not start at depot {}'.format('FAIL', k, depot))
            errors += 1
    return errors
        
def check_constraint3(solution, depot, num_vehicles, verbose):
    if verbose:
        print('\nChecking third constraint:')
    total_errors = 0
    for k in range(num_vehicles):
        errors = 0
        for s in solution[k]:
            leaving = 0
            leaving_tuples = []
            if s[1] != depot:
                for t in solution[k]:
                    if s[1] == t[0] and t[0]!=t[1]:
                        leaving += 1
                        leaving_tuples+=[t]
                if leaving == 0:
                    errors += 1
                    total_errors += 1
                    if verbose:
                        print('   {} -> Vehicle {} arrives at {} but never leaves - {}'.format('FAIL', k, s[1], s))
                if leaving > 1:
                    errors += 1
                    total_errors += 1
                    if verbose:
                        print('   {} -> Vehicle {} arrives at {} and leaves {} times {}'
                              .format('FAIL', k, s[1], leaving, leaving_tuples))
        if errors == 0:
            if verbose:
                print('   {} -> Vehicle {} has no dead ends'.format('OK', k))
    return total_errors
            
def check_constraint3_2(solution, depot, num_vehicles, verbose):
    if verbose:
        print('\nChecking third constraint (part 2):')
    total_errors = 0
    for k in range(num_vehicles):
        errors = 0
        for s in solution[k]:
            arriving = 0
            arriving_tuples = []
            if s[0] != depot:
                for t in solution[k]:
                    if s[0] == t[1] and t[0]!=t[1]:
                        arriving += 1
                        arriving_tuples+=[t]
                if arriving == 0:
                    errors += 1
                    total_errors += 1
                    if verbose:
                        print('   {} -> Vehicle {} never arrives at {} but leaves - {}'.format('FAIL', k, s[0], s))
                if arriving > 1:
                    errors += 1
                    total_errors += 1
                    if verbose:
                        print('   {} -> Vehicle {} never arrives at {} and leaves {} times {}'
                              .format('FAIL', k, s[0], arriving, arriving_tuples))
        if errors == 0:
            if verbose:
                print('   {} -> Vehicle {} has no impossible starts'.format('OK', k))
    return total_errors
    
def check_constraint4(solution, depot, num_vehicles, verbose):
    if verbose:
        print('\nChecking fourth constraint:')
    errors = 0
    for k in range(num_vehicles):
        ends_at_depot = 0
        for s in solution[k]:
            if s[1]==depot:
                ends_at_depot += 1
        if ends_at_depot:
            if verbose:
                print('   {} -> vehicle {} ends at depot {}'.format('OK', k, depot))
        else:
            errors += 1
            if verbose:
                print('   {} -> vehicle {} does not end at depot {}'.format('FAIL', k, depot))
    return errors
            
def check_constraint5(solution, num_vehicles, verbose):
    if verbose:
        print('\nChecking fifth constraint:')
    errors = 0
    for k in range(num_vehicles):
        route_map = {}

        for pair in solution[k]:
            route_map[pair[0]] = pair[1]

        visited = set()
        routes = []
        strRoute = ""
        try:
            for n in route_map.keys():
                run = True
                node = n
                if node not in visited:
                    while run:
                        if node in visited:
                            routes += [strRoute+str(node)]
                            strRoute = ""
                            run=False
                            break
                        strRoute += str(node)+"->"
                        visited.add(node)
                        node = route_map[node]
            if len(routes)==1:
                if verbose:
                    print('   {} -> No impossible loops found for vehicle {}: {}'.format('OK', k, routes[0]))
            else:
                errors += 1
                if verbose:
                    print('   {} -> Vehicle {} has impossible loops: {}'.format('FAIL', k, routes[1:]))
        except:
            print('Exception found!')
            errors += 1
    return errors
                
def check_capacity_errors(solution, customer_demand, vehicle_capacity, depot, verbose):
    if verbose:
        print('\nChecking route load and vehicle capacity:')
    errors = 0
    try:
        for k in range(len(vehicle_capacity)):
            route_load = 0
            for s in solution[k]:
                if(s[1]!=depot):
                    route_load += customer_demand[s[1]-1]
            if route_load>vehicle_capacity[k]:
                errors += 1
                if verbose:
                    print('   {} -> Vehicle {} has capacity {} and customer load {}'.format('Fail', k, vehicle_capacity[k], route_load))
            else:
                if verbose:
                    print('   {} -> Vehicle {} has capacity {} and customer load {}'.format('OK', k, vehicle_capacity[k], route_load))
    except:
        print('Exception found!')
        errors += 1
    return errors
            
def check_solution(solution, starting_depot, ending_depot, 
                   customer_demand, vehicle_capacity, check_capacity=True, verbose=True):
    num_customers = len(customer_demand)
    num_vehicles = len(vehicle_capacity)
    errors = 0
    errors += check_constraint1(solution, num_customers, num_vehicles, verbose)
    errors += check_constraint2(solution, starting_depot, num_vehicles, verbose)
    errors += check_constraint3(solution, ending_depot, num_vehicles, verbose)
    errors += check_constraint3_2(solution, starting_depot, num_vehicles, verbose)
    errors += check_constraint4(solution, ending_depot, num_vehicles, verbose)
    errors += check_constraint5(solution, num_vehicles, verbose)
    if check_capacity:
        errors += check_capacity_errors(solution, customer_demand, vehicle_capacity, ending_depot, verbose)
    return errors
    