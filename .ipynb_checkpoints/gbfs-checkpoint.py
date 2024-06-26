import heapq
import matplotlib.pyplot as plt
import numpy as np
import time

# Definisikan peta
peta = [
    ['#', '#', '#', '#', '#', '#', 'S', '#'],
    ['#', '.', '.', '.', '.', '.', '.', '#'],
    ['#', '.', '#', '#', '.', '.', '.', '#'],
    ['#', '.', '.', '.', '.', '#', '.', '#'],
    ['#', '.', '#', '#', '.', '#', '.', '#'],
    ['#', '.', '.', '.', '.', '.', '.', '#'],
    ['#', '.', '.', '.', '.', '.', '.', '#'],
    ['#', '#', 'G', '#', '#', '#', '#', '#']
]

# Fungsi untuk menemukan posisi awal dan tujuan
def find(peta):
    start = goal = None
    for i in range(len(peta)):
        for j in range(len(peta[i])):
            if peta[i][j] == 'S':
                start = (i, j)
            elif peta[i][j] == 'G':
                goal = (i, j)
    return start, goal

# Fungsi heuristik (jarak Manhattan)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Fungsi untuk menjalankan algoritma GBFS
def gbfs(peta):
    start, goal = find(peta)
    if not start or not goal:
        return None

    rows, cols = len(peta), len(peta[0])
    visited = set()
    pq = [(0, start)]  # priority queue dengan elemen (heuristic, position)
    came_from = {start: None}

    while pq:
        _, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            break

        # Tetangga (atas, bawah, kiri, kanan)
        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and neighbor not in visited:
                if peta[neighbor[0]][neighbor[1]] != '#':
                    heapq.heappush(pq, (heuristic(neighbor, goal), neighbor))
                    if neighbor not in came_from:
                        came_from[neighbor] = current

    # Rekonstruksi jalur dari goal ke start
    path = []
    if goal in came_from:
        current = goal
        while current:
            path.append(current)
            current = came_from[current]
        path.reverse()

    return path

# Visualisasi peta dan jalur menggunakan matplotlib
def visualisasi(peta, path):
    peta_visual = np.zeros((len(peta), len(peta[0])))
    for i in range(len(peta)):
        for j in range(len(peta[i])):
            if peta[i][j] == '#':
                peta_visual[i][j] = 1  # Dinding

    for (x, y) in path:
        if peta[x][y] not in ['S', 'G']:
            peta_visual[x][y] = 0.5  # Jalur

    start, goal = find(peta)
    peta_visual[start[0]][start[1]] = 0.3  # Titik awal
    peta_visual[goal[0]][goal[1]] = 0.7  # Titik tujuan

    plt.imshow(peta_visual, cmap='Blues')
    plt.show()

# Fungsi untuk mencetak peta dengan rute terpendek
def rute_terpendek(peta, path):
    peta_copy = [row[:] for row in peta]  # Membuat salinan dari peta asli
    for (x, y) in path:
        if peta_copy[x][y] not in ['S', 'G']:
            peta_copy[x][y] = '-'  # Tandai rute terpendek dengan '-'
    for row in peta_copy:
        print(' '.join(row))

# Menjalankan algoritma dan mengukur waktu eksekusi
start_time = time.perf_counter_ns()
path = gbfs(peta)
end_time = time.perf_counter_ns()
execution_time = end_time - start_time

if path:
    print(f"Jalur ditemukan: {path}")
    print(f"Jarak dari titik awal ke titik akhir sebanyak: {len(path) - 1} langkah")
    print(f"Kompleksitas waktu: {execution_time} ns")
    rute_terpendek(peta, path)  # Mencetak peta dengan rute terpendek
    visualisasi(peta, path)
else:
    print("Tidak ada jalur yang ditemukan.")
    print(f"Waktu eksekusi: {execution_time} ns")
