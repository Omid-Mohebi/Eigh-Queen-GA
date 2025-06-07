import random
import numpy as np

N_QUEENS = 8

def initialize_population_8queens(pop_size, n_queens):
    population = []
    for _ in range(pop_size):
        chromosome = np.random.randint(0, n_queens, n_queens)
        population.append(chromosome)
    return population

def calculate_fitness_8queens(chromosome):
    n_queens = len(chromosome)
    non_attacking_pairs = 0
    for i in range(n_queens):
        for j in range(i + 1, n_queens):
            if chromosome[i] != chromosome[j] and \
               abs(chromosome[i] - chromosome[j]) != abs(i - j):
                non_attacking_pairs += 1
    return non_attacking_pairs

def tournament_selection(population, fitness_values, tournament_size=3):
    selected_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_values[i] for i in selected_indices]
    winner_idx_in_tournament = np.argmax(tournament_fitness)
    winner_original_idx = selected_indices[winner_idx_in_tournament]
    return population[winner_original_idx].copy()

def crossover_8queens(parent1, parent2):
    n = len(parent1)
    if n <= 1:
        return parent1.copy(), parent2.copy()

    point1 = random.randint(0, n - 1)
    point2 = random.randint(point1, n - 1)
    if point1 > point2:
        point1, point2 = point2, point1
    elif point1 == point2 and n > 1:
        if point1 < n -1 :
            point2 = random.randint(point1 + 1, n - 1) if point1 < n-1 else point1
        elif point1 > 0:
            point1 = random.randint(0, point1-1) if point1 > 0 else point1

    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    offspring1[point1:point2+1] = parent2[point1:point2+1]
    offspring2[point1:point2+1] = parent1[point1:point2+1]

    return offspring1, offspring2


def mutate_8queens(chromosome, mutation_rate):
    if random.random() < mutation_rate:
        n_queens = len(chromosome)
        if n_queens == 0:
            return chromosome

        gene_to_mutate = random.randint(0, n_queens - 1)
        new_value = random.randint(0, n_queens - 1)
        chromosome[gene_to_mutate] = new_value
    return chromosome


def elitism_selection(population, fitness_values, elite_size):
    fitness_values_np = np.array(fitness_values)
    sorted_indices = np.argsort(fitness_values_np)[::-1]
    elite_indices = sorted_indices[:elite_size]
    return [population[i].copy() for i in elite_indices]

def print_board_8queens(chromosome):
    n_queens = len(chromosome)
    if n_queens == 0:
        print("Empty chromosome.")
        return
    for row in range(n_queens):
        line = ""
        for col in range(n_queens):
            if chromosome[col] == row:
                line += "Q  "
            else:
                line += ".  "
        print(line)
    print(f"Chromosome: {chromosome.tolist()}")
    print(f"Fitness (non-attacking pairs): {calculate_fitness_8queens(chromosome)}")


def ga_8queens_solver(pop_size=100, max_generations=500, n_queens=N_QUEENS,
                       crossover_rate=0.8, mutation_rate=0.1, elite_ratio=0.1,
                       tournament_size=3):
    population = initialize_population_8queens(pop_size, n_queens)
    elite_size = max(1, int(pop_size * elite_ratio))

    max_possible_fitness = n_queens * (n_queens - 1) // 2
    best_solution_overall = None
    best_fitness_overall = -1

    print(f"Starting GA for {n_queens}-Queens problem...")
    print(f"Max possible fitness (non-attacking pairs): {max_possible_fitness}\n")

    for generation in range(max_generations):
        fitness_values = [calculate_fitness_8queens(chrom) for chrom in population]

        current_best_fitness_in_gen = max(fitness_values)
        current_avg_fitness_in_gen = sum(fitness_values) / len(fitness_values)

        if current_best_fitness_in_gen > best_fitness_overall:
            best_fitness_overall = current_best_fitness_in_gen
            best_solution_overall = population[np.argmax(fitness_values)].copy()

        if (generation + 1) % 50 == 0 or generation == 0:
            print(f"Generation {generation + 1}/{max_generations}: "
                  f"Best Fitness in Gen = {current_best_fitness_in_gen}, "
                  f"Avg Fitness in Gen = {current_avg_fitness_in_gen:.2f}, "
                  f"Overall Best Fitness = {best_fitness_overall}")

        if best_fitness_overall == max_possible_fitness:
            print(f"\nSolution found in generation {generation + 1}!")
            print_board_8queens(best_solution_overall)
            return best_solution_overall, best_fitness_overall

        elite = elitism_selection(population, fitness_values, elite_size)
        new_population = elite[:]

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitness_values, tournament_size)
            parent2 = tournament_selection(population, fitness_values, tournament_size)

            if random.random() < crossover_rate:
                offspring1, offspring2 = crossover_8queens(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()

            offspring1 = mutate_8queens(offspring1, mutation_rate)
            offspring2 = mutate_8queens(offspring2, mutation_rate)

            new_population.append(offspring1)
            if len(new_population) < pop_size:
                new_population.append(offspring2)

        population = new_population[:pop_size]

    print("\nMaximum generations reached.")
    if best_solution_overall is not None:
        print("Best solution found during the run:")
        print_board_8queens(best_solution_overall)
    else:
        print("No solution found.")
    return best_solution_overall, best_fitness_overall

if __name__ == "__main__":
    best_chromosome, best_fitness = ga_8queens_solver(
        pop_size=300,
        max_generations=1000,
        n_queens=N_QUEENS,
        crossover_rate=0.9,
        mutation_rate=0.2,
        elite_ratio=0.1,
        tournament_size=5
    )

    if best_chromosome is not None:
        max_fit = N_QUEENS * (N_QUEENS - 1) // 2
        if best_fitness == max_fit:
            print(f"\nSuccessfully found a perfect solution for {N_QUEENS}-Queens!")
        else:
            print(f"\nCould not find a perfect solution. Best found had fitness {best_fitness}/{max_fit}.")