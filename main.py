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
    # Dosya kontrolü ekleyelim ki 2. dosya yoksa kod patlamasın
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
    # random.sample fonksiyonu tekrarsız (unique) seçim yapar.
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

    # 2. Şehir ID'lerini yan yana string haline getir
    # Örnek çıktı: "1 15 2 4 9 ..."
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

    # 2. Başlangıç şehrini seç
    if start_index >= len(unvisited): start_index = 0
    current_city = unvisited.pop(start_index)
    solution = [current_city]

    # 3. Tüm şehirler bitene kadar dön
    while unvisited:
        closest_city = None
        min_distance = float("inf")

        # --- ADIM A: Sadece EN YAKINI BUL (Döngü içinde işlem yapma) ---
        for candidate in unvisited:
            dist = calculate_distance(current_city, candidate)
            if dist < min_distance:
                min_distance = dist
                closest_city = candidate

        # --- ADIM B: Bulduktan sonra GİT ve LİSTEDEN SİL ---
        # (Bu kısım for döngüsünün DIŞINA (sola) alındı!)
        if closest_city:
            current_city = closest_city
            solution.append(current_city)
            unvisited.remove(current_city)
        else:
            break

    return solution


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
        print("0. Exit")
        print("=" * 50)

        choice = input("please select an option: ")
        if choice =="1":
            for file_name in files_to_test:
                print(f"\n{'-' * 10} TEST FİLE: {file_name} {'-' * 10}")

                # 1. PARSER TESTİ
                sehirler = parse_tsp_file(file_name)

                if not sehirler:
                    print(f"-> This file is being passed because {file_name} could not be loaded.")
                    continue

                # 2. MESAFE FONKSİYONU TESTİ (Madde 2)
                if len(sehirler) >= 2:
                    c1 = sehirler[0]
                    c2 = sehirler[1]
                    dist = calculate_distance(c1, c2)
                    print(f"Article 2 (Distance Test): Distance between City {c1.id} and City {c2.id} = {dist:.4f}")

                # 3. RASTGELE ÇÖZÜM VE TEKRAR KONTROLÜ (Madde 4 ve 4.a)
                random_sol = create_random_solution(sehirler)
                ids = [c.id for c in random_sol][:5]
                print(f"Item 4 (Random Solution - Top 5): {ids}")

                # Madde 4.a KONTROLÜ (Tekrar var mı? Tüm şehirler dahil mi?)
                unique_check = set(city.id for city in random_sol)
                if len(random_sol) == len(sehirler) and len(unique_check) == len(sehirler):
                    print(
                        "Article 4.a (Verification): PASSED. There is no repetition and all cities are available in the list.")
                else:
                    print("Article 4.a (Verification): ERROR! Missing city or repeat.")

                # 4. FITNESS HESAPLAMA
                fitness = calculate_fitness(random_sol)
                print(f"Fitness: {fitness:.4f}")

            print("\n=== ALL TESTS ARE COMPLETED ===")

        elif choice == "2":
            for file_name in files_to_test:
                print(f"\n File: {file_name}")
                sehirler = parse_tsp_file(file_name)

                if sehirler:
                    # 1. RANDOM TEST (Madde 6: Info)
                    print("\n[PART 2: Random Solution]")
                    random_sol = create_random_solution(sehirler)
                    print(f"Şehir Sayısı: {len(sehirler)}")
                    info(random_sol)

                    # 2. GREEDY TEST (Madde 7: Greedy vs Random)
                    print("\n[PART 3: Greedy vs Random]")

                    score_rand = calculate_fitness(random_sol)

                    greedy_sol = solve_greedy(sehirler, start_index=0)
                    score_greedy = calculate_fitness(greedy_sol)

                    print(f"(Random) Score: {score_rand:.4f}")
                    print(f"(Greedy) Score: {score_greedy:.4f}")

                    if score_greedy < score_rand:
                        diff = score_rand - score_greedy
                        print(f"BAŞARILI: Greedy algoritması {diff:.2f} puan daha iyi!")
                    else:
                        print("Random daha iyi çıktı (Çok nadir).")

                    print("-" * 30)
                else:
                    print("HATA: Dosya yüklenemedi.")

        elif choice == "0":
            print("Exiting the program ... Have a nice day")
            break
        else:
            print("!!! İnvalid selection")
