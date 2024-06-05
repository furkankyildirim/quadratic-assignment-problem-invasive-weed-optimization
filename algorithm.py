import numpy as np
from io import StringIO

class InvasiveWeedOptimization:
    """
    Quadratic Assignment Problem (QAP) using the Improved Weed Optimization (IWO) algorithm.

    Parameters
    ----------
    distances : list
        A list of lists containing the distance matrix.
    flows : list
        A list of lists containing the flow matrix.
    maximum_iterations : int
        The maximum number of iterations to run the algorithm.
    initial_population_size : int
        The initial population size.
    maximum_population_size : int
        The maximum population size.
    min_seed : int
        The minimum number of seeds.
    max_seed : int
        The maximum number of seeds.
    sigma_first : float
        The initial standard deviation value.
    sigma_final : float
        The final standard deviation value.
    search_area : tuple
        The search area for the seeds.
    m : int
        The dispersion reduction exponent. Default is 2.
    """

    max_iterations: int  # Maximum number of iterations
    initial_population_size: int  # Initial population size
    maximum_population_size: int  # Maximum population size
    min_seed: int  # Minimum number of seeds
    max_seed: int  # Maximum number of seeds
    sigma_init: float  # Initial standard deviation value
    sigma_final: float  # Final standard deviation value
    seach_area: tuple[int, int]  # Search area for the seeds
    m: int  # Dispersion reduction exponent (m)

    def __init__(self, size, distances, flows, max_iterations=20, m=2,
                 initial_population_size=15, maximum_population_size=20, min_seed=1, max_seed=3,
                 sigma_init=20, sigma_final=0.25, search_area=(-5, 5)):

        self.D = np.matrix(distances, dtype=object)  # Transportation cost matrix
        self.F = np.matrix(flows, dtype=object)  # Amount of resource units
        self.size = size

        # IWO parameters
        self.max_iterations = max_iterations
        self.initial_population_size = initial_population_size
        self.maximum_population_size = maximum_population_size
        self.min_seed = min_seed
        self.max_seed = max_seed
        self.m = m
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.search_area = search_area

    # Objective function
    def objective_function(self, p):
        cost = sum([self.F[i, j] * self.D[p[i], p[j]] for i in range(self.size) for j in range(self.size)])
        return cost

    # Main algorithm
    def run(self):
        # Initial population    
        population = []
        for i in range(self.initial_population_size):
            
            # Generate initial seed with uniform distribution
            rand = [np.random.uniform(self.search_area[0], self.search_area[1]) for _ in range(self.size)]
            seed = sorted(range(self.size), key=lambda idx: rand[idx])

            # Add to population, compute the objective function
            population.append((self.objective_function(seed), tuple(seed), tuple(rand)))

        # Sort the population reverse
        population.sort(reverse=True)

        # Perform calculations until the final number of operations is reached
        for t in range(0, self.max_iterations):

            # Update standard deviation using the formula
            sigma = (pow(((self.max_iterations - t) / self.max_iterations), self.m) * (self.sigma_init - self.sigma_final)) + self.sigma_final

            # Get best and worst solutions
            best_solution = min(population)[0]
            worst_solution = max(population)[0]

            # Reproduction phase
            for i in range(0, len(population)):

                # Calculate the number of seeds that this weed can produce
                ratio = float( (int(population[i][0]) - int(worst_solution)) / (int(best_solution) - int(worst_solution) + 1))
                s = (self.min_seed + (self.max_seed - self.min_seed) * ratio)

                # Each seed
                for _ in range(0, round(s)):
                    # Distribute around the parent plant
                    pos = np.array(population[i][2])
                    rand = pos + np.random.normal(0, sigma, self.size)

                    seed = sorted(range(self.size), key=lambda idx: rand[idx])

                    # Add to population, compute the objective function
                    population.append((self.objective_function(seed), tuple(seed), tuple(rand)))

            # Sort the population
            population.sort()

            # Remove the weak ones
            population = population[0:self.maximum_population_size]
        
        return population[0]
    
    def _matrix_to_string(self, matrix):
        output = StringIO()
        np.savetxt(output, matrix, fmt="%4d", delimiter=' ')
        return output.getvalue()


    def info(self):
        output = "Quadratic Assignment Problem (QAP) using the Improved Weed Optimization (IWO) algorithm.\n"
        output += f"\nTest Case:\n"
        output += f"    - Size: {self.size}\n"
        output += f"    - Distance matrix:\n"
        output += self._matrix_to_string(self.D)
        output += f"    - Flow matrix:\n"
        output += self._matrix_to_string(self.F)
        
        output += f"\nParameters:\n"
        output += f"    - Maximum iterations: {self.max_iterations}\n"
        output += f"    - Initial population size: {self.initial_population_size}\n"
        output += f"    - Maximum population size: {self.maximum_population_size}\n"
        output += f"    - Minimum seeds: {self.min_seed}\n"
        output += f"    - Maximum seeds: {self.max_seed}\n"
        output += f"    - Sigma init: {self.sigma_init}\n"
        output += f"    - Sigma final: {self.sigma_final}\n"
        output += f"    - Search area: {self.search_area}\n"

        return output