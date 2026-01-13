import math
import random
import os
import statistics

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    print("WARNING: matplotlib not installed. Graphs will not be shown.")
    HAS_MATPLOTLIB = False

# --- MADDE 1: Data Structure (Class Structure) ---
class City:
    def __init__(self, id, x, y):
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
        with open(filename, 'r') as file: #read the file
            lines = file.readlines() # read the all of lines
            parsing = False #close the parse
            for line in lines:
                if "NODE_COORD_SECTION" in line: # if see that command any line start
                    parsing = True
                    continue
                if "EOF" in line: # if see that command any line stop
                    break

                if parsing:
                    parts = line.strip().split() # split the element 1. count 2.x coord 3. y coord
                    if len(parts) >= 3:
                        new_city = City(parts[0], parts[1], parts[2]) #parts[0] parts[1] parts[2] = count x coord y coord
                        cities.append(new_city) # added to cities list to new city
        print(f"SUCCESSFUL: Loaded {len(cities)} cities from file {filename}.")
        return cities
    except Exception as e: #something wrong print error
        print(f"ERROR: {e}")
        return []

# --- MADDE 2: Distance Function ---
def calculate_distance(city1, city2):
    x_farki = city1.x - city2.x
    y_farki = city1.y - city2.y
    distance = math.sqrt(x_farki ** 2 + y_farki ** 2) #Pythagorean theorem
    return distance

# --- MADDE 3 & 4: Solution Storage & Random Solution ---
def create_random_solution(cities):
    random_solution = random.sample(cities, len(cities)) #random.sample means Permutation the random cities and cities lenght
    return random_solution

# --- EXTRA (PART 2): Fitness Calculation ---
def calculate_fitness(solution):
    total_distance = 0
    num_cities = len(solution)

    for i in range(num_cities):
        city_start = solution[i]
        city_end = solution[(i + 1) % num_cities]  #Returning from the last city of the list to the first city
        total_distance += calculate_distance(city_start, city_end) #Cumulative sum
    return total_distance

#info
def info(solution):
    score = calculate_fitness(solution) #total distance = score

    #element id convert int to str
    route_ids = [str(city.id) for city in solution]
    route_str = " ".join(route_ids) #adding space in the result (2 " " 3 " " 4)

    print(f"Solution Route: {route_str}")
    print(f"Score(fitness): {score:.4f}") #:.4f = 0.1234 -how many number show after the dot?

#greedy
def solve_greedy(cities, start_index=0):
    """
    Greedy Algorithm (Nearest Neighbor):
    Starts at a chosen node and always picks the closest unvisited city next.
    """
    if not cities: return []

    #1. Make a copy to preserve the original list.
    unvisited = cities.copy()

    # 2. choose your start city
    if start_index >= len(unvisited): start_index = 0
    current_city = unvisited.pop(start_index) # .pop You're completely removing that city from the equation.
    solution = [current_city]

    # 3. Turn around until all the cities are gone.
    while unvisited:
        closest_city = None
        min_distance = float("inf")

        # --- Part A:Just find the NEAREST one. ---
        for candidate in unvisited:
            dist = calculate_distance(current_city, candidate)
            if dist < min_distance: # Daha iyisini bulduk mu
                min_distance = dist #Evet bulduk, o zaman yeni rekor bu olsun
                closest_city = candidate

        # --- Part B: Once you find it, GO and DELETE IT FROM THE LIST. ---
        if closest_city:
            current_city = closest_city
            solution.append(current_city)
            unvisited.remove(current_city)
        else:
            break

    return solution

def create_population(cities , population_size):
    population = []
    for _ in range(population_size):# '_' değişklen önemsiz sadece döngünün dönmesini istiyoruz
        solution = create_random_solution(cities)
        population.append(solution)
    return population

"""
The `create_initial_population` function allows 
me to add a specific number of Greedy individuals to the population.
"""
def create_initial_population(cities,pop_size,greedy_count = 0):
    population = []
    for i in range(greedy_count):
        start_node_index = i%len(cities) #i%len(cities)   greedycount % len cities
        greedy_sol = solve_greedy(cities,start_index = start_node_index)
        population.append(greedy_sol)

    remaining_count = pop_size - greedy_count
    if remaining_count <0 : remaining_count = 0
    # If we tell it to generate -5 solutions, it can't generate -5, so it will generate 0.

    for _ in range(remaining_count):
        random_sol = create_random_solution(cities)
        population.append(random_sol)
    return population


# --- MADDE 13: Population Information ---
def info_population(population):
    if not population:
        print("Population is empty.")
        return
    # 1. Calculate everyone's fitness score.
    scores = [calculate_fitness(sol) for sol in population]

    # 2. Extract the statistics.
    best_score = min(scores)
    worst_score = max(scores)
    avg_score=sum(scores)/len(scores)

    # 3. Median calculation (Middle value)
    sorted_scores = sorted(scores)
    mid_index = len(scores) // 2 # 10//2 = 5     10%2 = 0     We need the index in the middle
    median_score = sorted_scores[mid_index]

    print(f"Population Size : {len(population)}")
    print(f"Best Score : {best_score:.4f}")
    print(f"Worst Score : {worst_score:4f}")
    print(f"Average Score : {avg_score:.4f}")
    print(f"Median Score : {median_score:.4f}")

#task 14
def tournament_selection(population,tournament_size=5):
    #1. Select candidates randomly.
    #This is a precaution to prevent errors if the population size is smaller than the tournament size.
    k = min(len(population),tournament_size)
    candidates = random.sample(population,k)

    #2. Find the best (shortest distance) candidate among the options.
    best_candidate = None
    best_score = float('inf')  #float('inf') == positive infinite

    for candidate in candidates:
        score = calculate_fitness(candidate)
        if score < best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate

def ordered_crossover(parent1,parent2):
    size =len(parent1)
    cut1,cut2 = sorted(random.sample(range(size),2)) # sorted ignore the error to code
    #cut1 8 cut2 2 is god is still work but cut1 2 cut2 8 it will be work to empty list

    child = [None] * size  # child = [ NONE ,NONE ,NONE ,NONE] how many size

    child[cut1:cut2] = parent1[cut1:cut2]

    current_ids = set(city.id for city in child[cut1:cut2])

    p2_index = 0
    for i in range(size):
        if child[i] is None:
            while p2_index < size and parent2[p2_index].id in current_ids:
                p2_index += 1

            if p2_index < size:
                child[i] = parent2[p2_index]
                current_ids.add(parent2[p2_index].id)

    return child

def inversion_mutation(solution,mutation_rate=0.1):
    #Roll the dice (a random number between 0.0 and 1.0)
    # mutation_rate = there is a 10% chance that we will make a random change (mutation) in its genes.
    if random.random() < mutation_rate:
        #Make a copy so as not to alter the original list.
        mutated_sol = solution[:]
        # 2. Select two random breakpoints.
        size = len(solution)
        idx1 , idx2 = random.sample(range(size),2)
        start,end = min(idx1,idx2),max(idx1,idx2)

        # 3. Invert the intermediate part (Inversion)
        # In Python, [::-1] is the inversion operation.
        segment = mutated_sol[start:end+1]
        mutated_sol[start:end+1] = segment[::-1]

        return mutated_sol
    # If the probability didn't work, send it back as is.
    return solution

def create_new_generation(previous_population,mutation_rate=0.1,crossover_rate=0.8):
    #previous_population = we select parent(mom and dad)
    #crossover_rate = Theres an 80% chance we'll mix the parents genes and create a hybrid child. 20% same genes to parent
    #mutation_rate = After the child is created, there is a 10% chance that we will make a random change (mutation) in its genes.
    new_population = []
    pop_size = len(previous_population)

    while len(new_population) < pop_size:
        parent1 = tournament_selection(previous_population , tournament_size=5)
        parent2 = tournament_selection(previous_population , tournament_size=5)
        #Select 5 people from the population and choose the one
        # with the shortest path (best) and assign it as "Parent".

        #selection
        if random.random() < crossover_rate:
            child = ordered_crossover(parent1,parent2)
        else:
            child = parent1[:] # [:] cloning to the list
            #If I hadn’t used [:], the mutations I made on the child would also have changed the parent.

        #crossover
        if None in child:
            continue

        #mutation
        child= inversion_mutation(child,mutation_rate)


        if len(child) != len(parent1):
            continue

        new_population.append(child)
    return new_population

def solve_tsp_genetic(cities, pop_size=100, iterations=3000, greedy_count=20, mutation_rate=0.1, crossover_rate=0.85):
    #pop_size =number of pirates searching for treasure
    #iterations = Epoch

    current_pop = create_initial_population(cities, pop_size, greedy_count)

    best_solution = min(current_pop, key=calculate_fitness) #`key` tells you which field you will be ranked by.
    best_score = calculate_fitness(best_solution)
    fitness_history = [best_score]

    for i in range(1, iterations + 1):

        current_pop = create_new_generation(current_pop, mutation_rate, crossover_rate)

        current_best = min(current_pop, key=calculate_fitness)
        current_score = calculate_fitness(current_best)

        if current_score < best_score:
            best_score = current_score
            best_solution = current_best[:]

        fitness_history.append(best_score)

    return best_solution, best_score, fitness_history

def plot_results(history , best_route):
    if not HAS_MATPLOTLIB:
        print("Matplotlib is missing. Skipping graphs.")
        return

    #1. Graph: Learning Curve (Score vs Epoch)
    plt.figure(figsize=(12,5)) #like a empyt paper 12,5 size

    # Left side: Scoreboard
    plt.subplot(1,2,1) #row column sequence number
    plt.plot(history)
    plt.title("Genetic Algorithm Convergence")
    plt.xlabel("Epoch(Generation)")
    plt.ylabel("Best Score (Distance)")
    plt.grid(True) #Add grid lines behind the graph.

    #2.Graphic: Route Map (Cities and Roads)
    plt.subplot(1,2,2)

    #Get the coordinates of the cities.
    x_coords = [city.x for city in best_route]
    y_coords = [city.y for city in best_route]

    #To close the road, add the starting city to the end.
    x_coords.append(best_route[0].x)
    y_coords.append(best_route[0].y)

    plt.plot(x_coords,y_coords,'o-r') #'o-r' redline with dots
    plt.title(f"Best Route Found (score : {calculate_fitness(best_route):.2f})")

    for city in best_route:
        plt.annotate(str(city.id), (city.x, city.y)) #Write a note/sticker on the dot.

    plt.tight_layout() #Set everything to automatic, ensure text doesn't overlap, and make sure margins are neat.
    plt.show()

# --- TASK 20
def run_parameter_comparison(cities):
    if not HAS_MATPLOTLIB:
        print("Matplotlib is missing. Skipping graphs.")
        return

    scenarios = {"Low Mutation (1%)": 0.01, "Normal Mutation (10%)": 0.1, "High Mutation (50%)": 0.5}
    plt.figure(figsize=(10, 6))
    print("\n--- COMPARISON STARTED ---")
    for name, m_rate in scenarios.items(): # name Low Mutation (1%)| m_rate 0.01
        print(f"Running scenario: {name} ...")

        _, final_score, history = solve_tsp_genetic(
            cities,
            pop_size=100,
            iterations=1500,
            greedy_count=10,
            mutation_rate=m_rate
        )
        # add Graf
        plt.plot(history, label=f"{name} (Score: {final_score:.0f})")

    # graffic settings
    plt.title("Impact of Mutation Rate on Convergence")
    plt.xlabel("Epochs")
    plt.ylabel("Best Score (Distance)")
    plt.legend() # It's the box in the corner of the graph that says "Which color line is which?".
    plt.grid(True)

    print("Displaying Comparison Chart...")
    plt.show()


def generate_final_report_stats(cities):
    print("\n" + "=" * 60)
    print("      PART 3: FINAL STATISTICAL REPORT GENERATION")
    print("=" * 60)
    print(f"Analyzing {len(cities)} cities...")

    # ---------------------------------------------------------
    # 1. RANDOM SEARCH (1000 RUNS)
    # ---------------------------------------------------------
    print("\n1. Running RANDOM SEARCH (1000 tests)...")
    random_scores = []
    for _ in range(1000):
        sol = create_random_solution(cities)
        random_scores.append(calculate_fitness(sol))

    r_best = min(random_scores)
    r_mean = statistics.mean(random_scores) # You made 1000 attempts. You add up all the scores and divide by 1000.
    r_stdev = statistics.stdev(random_scores) #Standard Deviation  It shows how much the data deviates from the average.
    r_variance = statistics.variance(random_scores) # It is the square of the standard deviation (s^2).

    print(f"   Done. Best Random: {r_best:.2f}")

    # ---------------------------------------------------------
    # 2. GREEDY ALGORITHM (ALL POSSIBLE STARTS)
    # ---------------------------------------------------------
    print("\n2. Running GREEDY ALGORITHM (All start nodes)...")
    greedy_scores = []
    for i in range(len(cities)):
        # Her şehirden başlatıp sonucu kaydediyoruz
        sol = solve_greedy(cities, start_index=i)
        greedy_scores.append(calculate_fitness(sol))

    g_best_5 = sorted(greedy_scores)[:5]  # En iyi 5 sonucu al
    g_mean = statistics.mean(greedy_scores)
    g_stdev = statistics.stdev(greedy_scores)
    g_variance = statistics.variance(greedy_scores)

    print(f"   Done. Best Greedy: {g_best_5[0]:.2f}")

    # ---------------------------------------------------------
    # 3. GENETIC ALGORITHM (10 RUNS)
    # ---------------------------------------------------------
    print("\n3. Running GENETIC ALGORITHM (10 Runs - This may take time)...")
    ga_scores = []

    best_params = {"pop_size": 150, "iterations": 2000, "greedy_count": 20, "mutation_rate": 0.1, "crossover_rate": 0.85}

    for run in range(1, 11):
        print(f"   Run {run}/10...", end="\r")
        _, best_score, _ = solve_tsp_genetic(cities, **best_params)
        ga_scores.append(best_score)

    print(f"   Done. Best GA: {min(ga_scores):.2f}            ")

    ga_mean = statistics.mean(ga_scores)
    ga_stdev = statistics.stdev(ga_scores)
    ga_variance = statistics.variance(ga_scores)

    # ---------------------------------------------------------
    # 4. PRINTING THE TABLE
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'METRIC':<25} | {'RANDOM (1000)':<15} | {'GREEDY (All)':<15} | {'GENETIC (10)':<15}")
    print("-" * 70)
    print(f"{'Best Score':<25} | {r_best:<15.2f} | {g_best_5[0]:<15.2f} | {min(ga_scores):<15.2f}")
    print(f"{'Mean (Average)':<25} | {r_mean:<15.2f} | {g_mean:<15.2f} | {ga_mean:<15.2f}")
    print(f"{'Standard Deviation':<25} | {r_stdev:<15.2f} | {g_stdev:<15.2f} | {ga_stdev:<15.2f}")
    print(f"{'Variance':<25} | {r_variance:<15.2f} | {g_variance:<15.2f} | {ga_variance:<15.2f}")
    print("-" * 70)

    print("\n>>> DETAILED RESULTS FOR REPORT:")
    print("Greedy Best 5 Results:", [f"{s:.2f}" for s in g_best_5])
    print("Genetic Algorithm 10 Runs:", [f"{s:.2f}" for s in ga_scores])
    print("=" * 70)


# --- MAIN BLOCK: TESTING REQUIREMENTS ---
if __name__ == "__main__":

    # Madde 1.c: Test on at least two different files
    files_to_test = ["berlin11_modified.tsp", "berlin52.tsp" ,"kroA100.tsp","kroA150.tsp"]
    current_cities = []
    selected_filename = "none"
    while True:
        print("\n" + "=" * 50)
        print("          Project Control MENU")
        print("=" * 50)
        print("1. Part 1")
        print("2. Part 2")
        print("3. Part 3")
        print("4. Part 4")
        print("5. Part 5")
        print("9. Part FINAL REPORT DATA GENERATOR")
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
                            current_greedy = solve_greedy(sehirler, start_index=i) #start_index = i It tries everything
                            # in turn as a starting point.
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
            print("\n" + "-" * 50)
            print("     PART 3: POPULATION & GENETIC ALGORITHM")
            print("-" * 50)
            print("1. Run Full Test (Tasks 11, 12, 13, 14, 15)")
            print("0. Return to Main Menu")

            suf_choice = input("Please select an option: ")

            if suf_choice == "1":
                print("\n[PART 3 INTEGRATED TEST STARTING...]")

                for file_name in files_to_test:
                    sehirler = parse_tsp_file(file_name)
                    if not sehirler: continue

                    print(f"\n" + "=" * 40)
                    print(f" FILE: {file_name}")
                    print("=" * 40)

                    # --- ARTİCLE 11 & 12: Population creater ---
                    print(">> [Task 11-12] Creating Initial Population...")
                    # 50 kişilik nüfus, 5 tanesi Greedy (Zeki), 45 tanesi Random.
                    my_pop = create_initial_population(sehirler, pop_size=50, greedy_count=5)

                    # Article 11
                    print(f"   Population Type: {type(my_pop)} (Correct)")
                    print(f"   Population Size: {len(my_pop)}")

                    # --- Article 13: Statictic (Info) ---
                    print("\n>> [Task 13] Population Statistics:")
                    info_population(my_pop)
                    # Bu fonksiyon Best, Worst, Average, Median basacak.

                    # --- Article 14: ---
                    print("\n>> [Task 14] Tournament Selection Test:")
                    # show the tournament system working
                    print("   Running selection 5 times to see who wins...")

                    parent1 = tournament_selection(my_pop,5)
                    parent2 = tournament_selection(my_pop,5)

                    score_p1 =calculate_fitness(parent1)
                    score_p2 =calculate_fitness(parent2)

                    print(f"   Parent 1 Score: {score_p1:.2f}")
                    print(f"   Parent 2 Score: {score_p2:.2f}")

                    #--- Crossover 15. artc
                    print("\n>> [task15] crossover (Ordered Crossover)")
                    # We breed two parents and produce a child.
                    child = ordered_crossover(parent1, parent2)
                    score_child = calculate_fitness(child)
                    # Let's check the child (Is there a mistake? Is the number of cities correct?)
                    print(f"   {'Parent 1 Score':<20} : {score_p1:.4f}")
                    print(f"   {'Parent 2 Score':<20} : {score_p2:.4f}")
                    print(f"   {'Child Score':<20} : {score_child:.4f}")
                    print("-" * 45)

                    best_parent = min(score_p1, score_p2)
                    if score_child < best_parent:
                        diff = best_parent - score_child
                        print(f"   RESULT: Child scored {diff:.2f} points better than parents.!")
                    else:
                        diff = score_child - best_parent
                        print(f"   RESULT: Normal. Child is {diff:.2f} points worse than parents..")
                        print("    (This is an expected situation and can be corrected with mutation.)")

                    print("\n   Test Finished for this file.")


        elif choice == "4":
            print("\n" + "-" * 50)
            print("     PART 4: MUTATION")
            print("-" * 50)
            print("1. Test Mutation (Task 16)")
            print("2. Test One Epoch (Jump from Generation 0 to 1) (task 17)")
            print("3. Final Genetic Algorithm (Task 18)")
            print("0. Return to Main Menu")
            sub_choice = input("Select: ")
            if sub_choice == "1":
                print("\n[ARTICLE 16: Mutation Test]")

                for file_name in files_to_test:
                    sehirler = parse_tsp_file(file_name)
                    if not sehirler: continue
                    print(f"\n>>> File: {file_name}")

                    pop = create_initial_population(sehirler, pop_size=50, greedy_count=5)
                    p1=tournament_selection(pop,5)
                    p2=tournament_selection(pop,5)
                    child=ordered_crossover(p1,p2)

                    score_before = calculate_fitness(child)
                    print(f"Child Score (Before Mutation): {score_before:.2f}")

                    #16.article
                    print("Applying Mutation(Rate: 1.0 -> %100 Definite Mutation) . . .")
                    mutated_child = inversion_mutation(child , mutation_rate=1.0)
                    score_after = calculate_fitness(mutated_child)
                    print(f"Child Score (after Mutation): {score_after:.2f}")

                    if child != mutated_child:
                        print("SUCCESS: Mutation changed the route structure! (Confirmed)")
                        if score_before != score_after:
                            diff = score_before - score_after
                            if diff > 0:
                                print(f"-> Result: Improved by {diff:.2f}")
                            else:
                                print(f"-> Result: Worse by {abs(diff):.2f}")
                        else:
                            print("-> Result: Score remained same (Coincedence), but route changed.")
                    else:
                        print("FAIL: The list did not change at all.")

            if sub_choice == "2":
                print("\n[ARTICLE 17: One Epoch Test]")
                for file_name in files_to_test:
                    sehirler = parse_tsp_file(file_name)
                    if not sehirler: continue
                    print(f"\n>>> File: {file_name}")

                    #1. EPOCH 0: Initial Population
                    print("Creating Generation 0 (Initial) . . .")
                    pop_gen0 = create_initial_population(sehirler,50,5) #greedy 5 ,45 random

                    print("\n--- Generation 0 stats ---")
                    info_population(pop_gen0)

                    #Let's save the best score for comparison.
                    best_gen0 = min([calculate_fitness(s) for s in pop_gen0])

                    # 2. EPOCH 1: Evolutionary Transition (Function of Article 17)
                    print("\nRunning Evolution (creating Generation1). . . ")
                    # Let's assume the crossover rate is 80% and the mutation rate is 10%.
                    pop_gen1 = create_new_generation(pop_gen0 , mutation_rate=0.1, crossover_rate=0.8)

                    print("\n--- GENERATION 1 STATS ---")
                    info_population(pop_gen1)

                    best_gen1 = min([calculate_fitness(s) for s in pop_gen1])
                    #result
                    print("-" *40)
                    if len(pop_gen1) == len(pop_gen0):
                        print("SUCCESS: Population size preserved.")
                    else:
                        print("FAIL: Population size changed!")

                    diff = best_gen0 - best_gen1
                    if diff > 0 :
                        print(f"EVOLUTION WORKING: Best score omproved by {diff:.2f} points.")
                    elif diff < 0:
                        print(f"WARNING: Best score got worse by {abs(diff):.2f} points.")
                        print("(This happens sometimes in early generations without Elitism. Keep going!)")
                    else:
                        print("STAGNATION: Best score remained exactly the same.")

            if sub_choice == "3":
                print("\n" + "=" * 50)
                print("     FINAL RUN AND GRAPHICS")
                print("=" * 50)

                for file_name in files_to_test:
                    sehirler = parse_tsp_file(file_name)
                    if not sehirler: continue
                    print(f"\n>>> File: {file_name}")

                    # Calculate your Greedy Score for reference
                    greedy_sol = solve_greedy(sehirler, 0)
                    greedy_score = calculate_fitness(greedy_sol)
                    print(f"Greedy Score to beat {greedy_score:.4f}")
                    print("Running Genetic Algorithm . . .")

                    #starting algorthm
                    final_route , final_score ,history = solve_tsp_genetic(sehirler,pop_size=100 , iterations=3000,greedy_count=10)
                    print("-" * 30)
                    print(f"FINAL RESULT ({file_name}):")
                    print(f"Greedy Score : {greedy_score:.4f}")
                    print(f"Genetic Score: {final_score:.4f}")

                    diff = greedy_score - final_score
                    if  diff > 0 :
                        print(f"SUCCESS: Genetic Algorithm is better by {diff:.2f} points!")
                    else:
                        print(f"RESULT: Genetic is worse by {abs(diff):.2f} points. (Try increasing iterations)")
                    print("-" * 30)

                    print("Displaying Graphs")
                    plot_results(history , final_route)

        elif choice == "5":
            print("\n" + "=" * 50)
            print("     PART 5: COMPARE PARAMETERS")
            print("=" * 50)

            for file_name in files_to_test:
                print(f"Loading {file_name} for comparison...")
                sehirler = parse_tsp_file(file_name)

                if sehirler:
                    run_parameter_comparison(sehirler)

        elif choice == "9":
            print("\n" + "#" * 60)
            print("     PART 3: FINAL STATISTICAL REPORT (ALL FILES)")
            print("#" * 60)

            for file_name in files_to_test:
                print(f"\n\n>>> PROCESSING FILE: {file_name} <<<")
                sehirler = parse_tsp_file(file_name)

                if not sehirler:
                    print(f"Skipping {file_name} (Could not load).")
                    continue

                generate_final_report_stats(sehirler)

                print(f">>> REPORT FOR {file_name} COMPLETED.")
                print("-" * 60)

        elif choice == "0":
            print("Exiting the program ... Have a nice day")
            break
        else:
            print("!!! İnvalid selection")