import math
import random
import os


# --- MADDE 1: Data Structure (Class Structure) ---
class City:
    def __init__(self, id, x, y):
        # Madde 1.b: Datatypes (int for ID, float for coordinates)
        self.id = int(id)
        self.x = float(x)
        self.y = float(y)

    def __repr__(self):
        return f"City(ID={self.id}, X={self.x}, Y={self.y})"


# --- MADDE 1: Parser Function ---
def parse_tsp_file(filename):
    cities = []
    if not os.path.exists(filename):
        print(f"WARNING: File '{filename}' not found.")
        return []

    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            parsing = False
            for line in lines:
                if "NODE_COORD_SECTION" in line:
                    parsing = True
                    continue
                if "EOF" in line:
                    break

                if parsing:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        new_city = City(parts[0], parts[1], parts[2])
                        cities.append(new_city)
        print(f"SUCCESSFUL: Loaded {len(cities)} cities from file {filename}.")
        return cities
    except Exception as e:
        print(f"ERROR: {e}")
        return []

# --- MADDE 2: Distance Function ---
def calculate_distance(city1, city2):
    """
    Returns float distance between two cities.
    """
    x_farki = city1.x - city2.x
    y_farki = city1.y - city2.y
    distance = math.sqrt(x_farki ** 2 + y_farki ** 2)
    return distance

# --- MADDE 3 & 4: Solution Storage & Random Solution ---
# Karar (Madde 3): Şehirleri tutmak için Python 'list' yapısı seçilmiştir.
def create_random_solution(cities):
    """
    Madde 4: Create a random one based on file.
    Uses random.sample to avoid repetition (Madde 4.a).
    """
    random_solution = random.sample(cities, len(cities))
    return random_solution

# --- EXTRA (PART 2): Fitness Calculation ---
def calculate_fitness(solution):
    total_distance = 0
    num_cities = len(solution)

    for i in range(num_cities):
        city_start = solution[i]
        city_end = solution[(i + 1) % num_cities]  # Döngüsel yol
        total_distance += calculate_distance(city_start, city_end)
    return total_distance

#info
def info(solution):
    score = calculate_fitness(solution)

    #converter to string
    route_ids = [str(city.id) for city in solution]
    route_str = " ".join(route_ids)

    print(f"Solution Route: {route_str}")
    print(f"Score(fitness): {score:.4f}")

#greedy
def solve_greedy(cities, start_index=0):
    """
    Greedy Algorithm (Nearest Neighbor):
    Starts at a chosen node and always picks the closest unvisited city next.
    """
    if not cities: return []

    # 1. Orijinal listeyi korumak için kopyasını al
    unvisited = cities.copy()

    # 2. choose your start city
    if start_index >= len(unvisited): start_index = 0
    current_city = unvisited.pop(start_index)
    solution = [current_city]

    # 3. Turn around until all the cities are gone.
    while unvisited:
        closest_city = None
        min_distance = float("inf")

        # --- ADIM A:Just find the NEAREST one. ---
        for candidate in unvisited:
            dist = calculate_distance(current_city, candidate)
            if dist < min_distance:
                min_distance = dist
                closest_city = candidate

        # --- ADIM B: Once you find it, GO and DELETE IT FROM THE LIST. ---
        if closest_city:
            current_city = closest_city
            solution.append(current_city)
            unvisited.remove(current_city)
        else:
            break

    return solution

def create_population(cities , population_size):
    population = []
    for _ in range(population_size):
        solution = create_random_solution(cities)
        population.append(solution)
    return population

def create_initial_population(cities,pop_size,greedy_count = 0):
    population = []
    for i in range(greedy_count):
        start_node_index = i%len(cities)
        greedy_sol = solve_greedy(cities,start_index = start_node_index)
        population.append(greedy_sol)

    remaining_count = pop_size - greedy_count
    if remaining_count <0 : remaining_count = 0

    for _ in range(remaining_count):
        random_sol = create_random_solution(cities)
        population.append(random_sol)
    return population

# --- MAIN BLOCK: TESTING REQUIREMENTS ---
if __name__ == "__main__":

    # Madde 1.c: Test on at least two different files
    files_to_test = ["berlin11_modified.tsp", "berlin52.tsp"]
    current_cities = []
    selected_filename = "none"
    while True:
        print("\n" + "=" * 50)
        print("          Project Control MENU")
        print("=" * 50)
        print("1. Part 1")
        print("2. Part 2")
        print("3. Part 3")
        print("0. Exit")
        print("=" * 50)

        choice = input("please select an option: ")
        if choice =="1":
            for file_name in files_to_test:
                print(f"\n{'-' * 10} TEST FİLE: {file_name} {'-' * 10}")

                # 1. PARSER TEST
                sehirler = parse_tsp_file(file_name)

                if not sehirler:
                    print(f"-> This file is being passed because {file_name} could not be loaded.")
                    continue

                # 2. Distance Func TEST 2.article
                if len(sehirler) >= 2:
                    c1 = sehirler[0]
                    c2 = sehirler[1]
                    dist = calculate_distance(c1, c2)
                    print(f"Article 2 (Distance Test): Distance between City {c1.id} and City {c2.id} = {dist:.4f}")

                # 3. RANDOM SOLUTION AND REPEAT CHECK (article 4 ve 4.a)
                random_sol = create_random_solution(sehirler)
                ids = [c.id for c in random_sol][:5]
                print(f"Item 4 (Random Solution - Top 5): {ids}")

                unique_check = set(city.id for city in random_sol)
                if len(random_sol) == len(sehirler) and len(unique_check) == len(sehirler):
                    print(
                        "Article 4.a (Verification): PASSED. There is no repetition and all cities are available in the list.")
                else:
                    print("Article 4.a (Verification): ERROR! Missing city or repeat.")

                # 4. FITNESS calculate
                fitness = calculate_fitness(random_sol)
                print(f"Fitness: {fitness:.4f}")

            print("\n=== ALL TESTS ARE COMPLETED ===")

        elif choice == "2":
            while True:
                print("1. (5. 6.)fitness and info functions")
                print("2. (7.)greedy vs random")
                print("3. (8.)Best Spawm point")
                print("4. (9.)POPULATION GENERATION ")
                print("0. Return to Main Menu")
                sub_choice = input("Part 2 please select an option: ")
                if sub_choice == "1":
                    print("\n[TEST: INFO & FITNESS]")
                    for file_name in files_to_test:
                        print(f"\nDosya: {file_name}")
                        sehirler = parse_tsp_file(file_name)
                        if sehirler:
                            rand_sol = create_random_solution(sehirler)
                            # info func article 6
                            info(rand_sol)

                    # ---  GREEDY vs RANDOM (art 7) ---
                elif sub_choice == "2":
                    print("\n[TEST: GREEDY vs RANDOM]")
                    for file_name in files_to_test:
                        print(f"\nFile: {file_name}")
                        sehirler = parse_tsp_file(file_name)
                        if sehirler:
                            # Random Score
                            r_sol = create_random_solution(sehirler)
                            r_score = calculate_fitness(r_sol)

                            #1 Greedy Score
                            g_sol = solve_greedy(sehirler, start_index=0)
                            g_score = calculate_fitness(g_sol)

                            print(f"Random Skor: {r_score:.4f}")
                            print(f"Greedy Skor: {g_score:.4f}")

                            if g_score < r_score:
                                print(f"-> Greedy algorithm {r_score - g_score:.2f} score is better!")

                    # --- ITERATIVE GREEDY (MADDE 8) ---
                elif sub_choice == "3":
                    print("\n[TEST: THE BEST STARTING POINT (ARTICLE 8)]")
                    for file_name in files_to_test:
                        print(f"\n>>> Fıle scanned: {file_name}")
                        sehirler = parse_tsp_file(file_name)
                        if not sehirler: continue

                        best_score = float('inf')
                        best_start_node = -1

                        for i in range(len(sehirler)):
                            current_greedy = solve_greedy(sehirler, start_index=i)
                            score = calculate_fitness(current_greedy)
                            print(f"\n--- Start Node Index: {i} ---")
                            info(current_greedy)

                            if score < best_score:
                                best_score = score
                                best_start_node = i

                        print(f"\n*** Result ({file_name}) ***")
                        print(f"Best Starting City Index: {best_start_node}")
                        print(f"Best Score: {best_score:.4f}")

                elif sub_choice =="4":
                    print("\n Art 9: POPULATION GENERATION")
                    sehirler = parse_tsp_file("berlin52.tsp")

                    if sehirler:
                        greedy_reference = solve_greedy(sehirler, 0)
                        greed_score = calculate_fitness(greedy_reference)
                        print(f"Greedy Score: {greed_score:.4f}")
                        print("-" * 30)

                        #1. 100 random solutution creater
                        saved_population = [] #reset list
                        best_random_score = float('inf')
                        print("100 random solution created")
                        for i in range(100):
                            sol = create_random_solution(sehirler)
                            fit = calculate_fitness(sol)
                            saved_population.append(sol)
                            if i <5:
                                print(f"Random Sol{i+1}: Fitness = {fit:.4f}")

                            if fit < best_random_score:
                                best_random_score = fit

                        print ("\n(The other 95 are hidden.")
                        print("-"  *30)
                        print(f"best random score: {best_random_score:.4f}")
                        print(f"Greedy Score: {greed_score:.4f}")
                        diff = best_random_score - greed_score
                        print(f"difference: {diff:.4f}\n")

                elif sub_choice == "0":
                    print("Exiting the part 2")
                    break

        elif choice == "3":
            print("\n" + "-" *40)
            print("     part3:Population And GA")
            print("-" *40)
            print("1. (11. 12.) Population with Greedy Seeding")
            print("0. Main Menu")

            suf_choice = input("Please select an option: ")

            if suf_choice == "1":
                print("\n[11. And 12. Arc: Population with Greedy Seeding]")
                for file_name in files_to_test:
                    sehirler = parse_tsp_file(file_name)
                    if not sehirler: continue
                    print(f"\nFile: {file_name}")
                    #TEST A: Random Only (Clause 12.a)
                    print("Test A : A completely random population of 50 people.")
                    pop_pure_random = create_initial_population(sehirler,50,0)
                    #Let's check the first person's fitness (It should be bad).
                    score_rnd = calculate_fitness(pop_pure_random[0])
                    print(f"Size : {len(pop_pure_random)}")
                    print(f"First Element Score : {score_rnd:.2f}")

                    # TEST B: Mixed (Item 12.b)
                    print("Test B : A population of 50 people (5 of whom are GREEDY)")
                    # Let's have 50 people, but 5 of them should come from the Greedy algorithm.
                    pop_mixed = create_initial_population(sehirler,50,5)

                    # Since the first element is greedy, its score should be much better.
                    score_greedy = calculate_fitness(pop_mixed[0])
                    print(f"Size : {len(pop_mixed)}")
                    print(f"First Element Score : {score_greedy:.2f}")

                    #part11
                    if score_greedy < score_rnd:
                        print(f"SUCCESSFUL: Populations containing Greedy start better. ")
                    #Create a population of 50 people.
                    my_population = create_population(sehirler, 50)
                    print(f"\nFile: {file_name}") #class'list'
                    print(f"Population Type: {type(my_population)}")
                    print(f"Population Size: {len(my_population)}")
                    print(f"First Solution Type: {type(my_population[0])}")
                    print("Test Successful!\n")

        elif choice == "0":
            print("Exiting the program ... Have a nice day")
            break
        else:
            print("!!! İnvalid selection")