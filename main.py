import os
import time
from algorithm import InvasiveWeedOptimization

max_iterations = [20, 25]
initial_population_size = [15, 20, 30]
maximum_population_size = [20, 25, 30]
min_seed = 1
max_seed = [3, 5]
sigma_init = 20
sigma_final = [0.25, 0.5]
search_area = [(-5, 5), (-10, 10), (-20, 20)]


# Read the files on data folder
files = os.listdir('data')

# Run the Invasive Weed Optimization algorithm for each file
for name in files:
    
    print(f'Running IWO for {name}...')

    # Open the file
    file = open('data/' + name, 'r')

    # Get size of the problem
    size = int(file.readline())
    print(f'Sample size: {size}')
    file.readline()

    # Read the Flow matrix
    F = [[int(i) for i in file.readline().split()] for _ in range(size)]
    file.readline()

    # Read the Distance matrix
    D = [[int(i) for i in file.readline().split()] for _ in range(size)]
    file.close()

    best_solution = float('inf')
    
    for max_iter in max_iterations:
        for init_pop in initial_population_size:
            for max_pop in maximum_population_size:
                for max_s in max_seed:
                    for sigma_f in sigma_final:
                        for s_area in search_area:
                            
                            solver = InvasiveWeedOptimization(size=size, distances=D, flows=F, max_iterations=max_iter, m=2,
                                                              initial_population_size=init_pop, maximum_population_size=max_pop, min_seed=min_seed, max_seed=max_s,
                                                              sigma_init=sigma_init, sigma_final=sigma_f, search_area=s_area)
                            
                            start = time.time()
                            population = solver.run()
                            end = time.time()

                            output = solver.info()
                            output += f"\nResults:\n"
                            output += f"    - Objective value: {population[0]}\n"
                            output += f"    - Execution time: {end - start} seconds\n"
                            output += f"    - Best solution: {population[1]}\n"

                            # Save the results
                            print(output)

                            if population[0] < best_solution:
                                best_solution = population[0]
                                with open(f'results/{name.split(".")[0]}_results.txt', 'w') as file:
                                    file.write(output)
                                    file.close()
    print(f'IWO for {name} finished!')
                           
                        

