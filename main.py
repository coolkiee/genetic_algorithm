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
    mid_index = len(scores) // 2
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
    best_score = float('inf')

    for candidate in candidates:
        score = calculate_fitness(candidate)
        if score < best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate

def ordered_crossover(parent1,parent2):
    size =len(parent1)
    #1. Rastgele iki kesme noktası belirle
    cut1,cut2 = sorted(random.sample(range(size),2))

    #2. Çocuğu boş (None) olarak başlat
    child = [None] * size

    # 3. Parent1'den parça kopyala (Genetik Miras 1)
    child[cut1:cut2] = parent1[cut1:cut2]

    # Kopyalanan şehirlerin ID'lerini bir sette tut (Hızlı kontrol için)
    current_ids = set(city.id for city in child[cut1:cut2])

    p2_index = 0
    for i in range(size):
        if child[i] is None:
            #parent2'deki sıradaki şehri bul  (zaten eklenmemiş olanı)
            while p2_index < size and parent2[p2_index].id in current_ids:
                p2_index += 1

            #şehri ekle
            if p2_index < size:
                child[i] = parent2[p2_index]
                current_ids.add(parent2[p2_index].id)

    return child

def inversion_mutation(solution,mutation_rate=0.1):
    #Roll the dice (a random number between 0.0 and 1.0)
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
    new_population = []
    pop_size = len(previous_population)




    while len(new_population) < pop_size:
        parent1 = tournament_selection(previous_population , tournament_size=5)
        parent2 = tournament_selection(previous_population , tournament_size=5)

        #selection
        if random.random() < crossover_rate:
            child = ordered_crossover(parent1,parent2)
        else:
            child = parent1[:]

        #crossover
        if None in child:
            continue

        #mutation
        child= inversion_mutation(child,mutation_rate)


        if len(child) != len(parent1):
            continue

        new_population.append(child)
    return new_population


def solve_tsp_genetic(cities, pop_size=100, iterations=3000, greedy_count=10):
    # 1. Create Initial Population
    # pop_size 100 ve greedy_count 10, Berlin52 için çok iyi bir başlangıçtır.
    current_pop = create_initial_population(cities, pop_size, greedy_count)

    # Global Best: En iyiyi hafızada tutuyoruz
    best_solution = min(current_pop, key=calculate_fitness)
    best_score = calculate_fitness(best_solution)

    print(f"Initial Best Score (Gen 0): {best_score:.4f}")

    # 2. Loop through Epochs
    for i in range(1, iterations + 1):

        # --- KRİTİK AYAR ---
        # Mutasyonu 0.4 yapma, çok yıkıcı.
        # Berlin52 için ideal oran 0.15 ile 0.20 arasıdır.
        # Crossover rate 0.8 veya 0.85 iyidir.

        current_pop = create_new_generation(current_pop, mutation_rate=0.15, crossover_rate=0.85)

        # Bu neslin en iyisini bul
        current_best = min(current_pop, key=calculate_fitness)
        current_score = calculate_fitness(current_best)

        # 3. Check if we found a new global best
        if current_score < best_score:
            best_score = current_score
            best_solution = current_best[:]  # Kopyasını sakla
            # Her iyileşmeyi görmek motivasyon artırır:
            print(f"Epoch {i}: New Best Found! Score = {best_score:.4f}")

        # 4. Progress report (Her 500 turda bir yazsın, ekranı doldurmasın)
        if i % 500 == 0:
            print(f"Epoch {i}/{iterations} completed. Current Best: {best_score:.4f}")

    return best_solution, best_score


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
        print("4. Part 4")
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

                # --- PART 3: GENETİK ALGORİTMA (Maddeler 11-14) ---

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

                    # --- MADDE 11 & 12: Popülasyon Oluşturma ---
                    # Hem liste yapısını (11) hem de Greedy desteğini (12) kullanıyoruz.
                    print(">> [Task 11-12] Creating Initial Population...")
                    # 50 kişilik nüfus, 5 tanesi Greedy (Zeki), 45 tanesi Random.
                    my_pop = create_initial_population(sehirler, pop_size=50, greedy_count=5)

                    # Madde 11 Kontrolü (Liste mi?)
                    print(f"   Population Type: {type(my_pop)} (Correct)")
                    print(f"   Population Size: {len(my_pop)}")

                    # --- MADDE 13: İstatistikler (Info) ---
                    print("\n>> [Task 13] Population Statistics:")
                    info_population(my_pop)
                    # Bu fonksiyon Best, Worst, Average, Median basacak.

                    # --- MADDE 14: Seçim (Selection) Testi ---
                    print("\n>> [Task 14] Tournament Selection Test:")
                    # Turnuva sisteminin çalıştığını kanıtlayalım
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
                print("     FINAL GENETIC ALGORITHM RUN")
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
                    final_route , final_score = solve_tsp_genetic(sehirler,pop_size=100 , iterations=3000,greedy_count=10)
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



        elif choice == "0":
            print("Exiting the program ... Have a nice day")
            break
        else:
            print("!!! İnvalid selection")