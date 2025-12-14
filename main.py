import math
import random

class City:
    def __init__(self, id,x,y):
        self.id = int(id)
        self.x = float(x)
        self.y = float(y)

    def __repr__(self):
        return f"City(ID={self.id}, X={self.x}, Y={self.y})"

def parse_tsp_file(filename):
    cities =[]

    try:
        with open(filename,'r') as file:
            lines = file.readlines()
            parsing = False

            for line in lines:
            #when it see ""NODE_COORD_SECTION" text its start the reading
                if "NODE_COORD_SECTION" in line:
                    parsing = True
                    continue
                #if you see "EOF" STOP
                if "EOF" in line:
                    break

                #if parsing mode activated get the numbers
                if parsing:
                    parts = line.strip().split()
                    if len(parts) >=3:
                        new_city = City(parts[0],parts[1],parts[2])
                        cities.append(new_city)
        print(f"{filename} {len(cities)}")
        return cities
    except FileNotFoundError:
        print(f"File {filename} not found")
        return []

def calculate_distance(city1,city2):
    x_farki = city1.x - city2.x
    y_farki = city1.y - city2.y
    distance = math.sqrt(x_farki**2 + y_farki**2)
    return distance

def create_random_solution(cities):
    """
    Takes the given list of cities and randomly shuffled
    Returns a new list (route).
    """
    # The random.sample function takes the list and creates a random copy of it.
    # The original 'cities' list is not corrupted.
    random_solution = random.sample(cities , len(cities))
    return random_solution


def calculate_fitness(solution):
    total_distance = 0
    num_cities = len(solution)

    for i in range(num_cities):
        city_start = solution[i]
        # Son şehirden ilk şehre dönüşü (Döngü) sağlamak için modülo (%) kullanıyoruz
        city_end = solution[(i + 1) % num_cities]

        # Madde 5'te istendiği gibi calculate_distance kullanılıyor
        total_distance += calculate_distance(city_start, city_end)

    return total_distance



if __name__ == "__main__":
    file_name ="berlin11_modified.tsp"
    sehirler = parse_tsp_file(file_name)

    if len(sehirler):
        print("\n--- KONTROL BAŞLIYOR ---")

        print("part1 2.article\n")
        c1 = sehirler[0]
        c2=sehirler[1]
        print(f"1. city: {c1}")
        print(f"2. city: {c2}")
        distance=calculate_distance(c1,c2)
        print(f"Distance: {distance:.4f}")

        print("part1 article 4\n")
        print(f"Original ranking(first 5 cities): {sehirler[:5]}")
        # random create
        random_solution = create_random_solution(sehirler)
        print(f"Random ranking(first 5 cities): {random_solution[:5]}")
        if sehirler[0] != random_solution[0] or sehirler[1] != random_solution[1]:
            print("\nSUCCESSFUL: List shuffled!.")
        else:
            print("\nNOTE: If by chance it came out the same or it didn't get mixed up, try again.")

        print("\npart2 5.article ")
        solution = calculate_fitness(random_solution)
        print(f"\nBu çözümün Fitness Değeri (Toplam Mesafe): {solution:.4f}")
        if solution > 0:
            print("Madde 5 Başarıyla Tamamlandı.")
    else:
        print("there is not have a enough cities")





