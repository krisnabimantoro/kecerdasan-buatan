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

# Fungsi untuk menjalankan algoritma A*
def a_star(peta):
    start, goal = find(peta)
    if not start or not goal:
        return None

    rows, cols = len(peta), len(peta[0])
    open_set = [(0, start)]  # priority queue dengan elemen (f_score, position)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        neighbors = [(current[0] + dx, current[1] + dy)
                     for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        for neighbor in neighbors:
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if peta[neighbor[0]][neighbor[1]] == '#':
                    continue
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + \
                        heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

# Visualisasi peta dan jalur menggunakan matplotlib
def viisualisasi(peta, path):
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
path = a_star(peta)
end_time = time.perf_counter_ns()
execution_time = end_time - start_time  # waktu eksekusi dalam nanodetik

if path:
    print(f"Jalur ditemukan: {path}")
    print(f"Jarak dari titik awal ke titik akhir sebanyak: {len(path) - 1} langkah")
    print(f"Waktu eksekusi: {execution_time} ns")
    rute_terpendek(peta, path)  # Mencetak peta dengan rute terpendek
    viisualisasi(peta, path)
else:
    print("Tidak ada jalur yang ditemukan.")
    print(f"Waktu eksekusi: {execution_time} ns")
