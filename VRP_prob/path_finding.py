import numpy as np
import matplotlib.pyplot as plt
from collections import deque

plt.rcParams['font.sans-serif'] = ['SimHei']   
plt.rcParams['axes.unicode_minus'] = False       

# ============================
# 1. 仓库布局与任务点定义
# ============================

# 定义仓库布局，0代表可通行区域，1代表货架
cache = np.zeros((28, 28), dtype=int)

cache[0, 1:8]   = 1
cache[0, 10:17] = 1
cache[0, 19:26] = 1

cache[3:5, 1:8]   = 1
cache[3:5, 10:17] = 1
cache[3:5, 19:26] = 1

cache[7:9, 1:8]   = 1
cache[7:9, 10:17] = 1
cache[7:9, 19:26] = 1

cache[11:13, 1:8]   = 1
cache[11:13, 10:17] = 1
cache[11:13, 19:26] = 1

cache[15:17, 1:8]   = 1
cache[15:17, 10:17] = 1
cache[15:17, 19:26] = 1

cache[19:21, 1:8]   = 1
cache[19:21, 10:17] = 1
cache[19:21, 19:26] = 1

cache[23:25, 1:8]   = 1
cache[23:25, 10:17] = 1
cache[23:25, 19:26] = 1

cache[27, 1:8]   = 1
cache[27, 10:17] = 1
cache[27, 19:26] = 1

# 定义任务点
picks = {
    0: [5, 14, 20, 24],
    3: [11, 14, 21, 23],
    4: [2, 5],
    7: [2],
    8: [2, 6, 11, 15, 20, 24],
    11: [4, 14, 21],
    12: [2, 6, 12, 14],
    15: [11, 13, 15, 22],
    16: [3, 6, 20, 22, 25],
    19: [5],
    20: [5, 13, 15, 20, 23],
    23: [3, 5, 15, 20, 24],
    24: [2, 11, 14],
    27: [3, 7, 11, 15, 21]
}

# 定义出发点
SEpoint = (0, 5)

tasks = []
for r, cols in picks.items():
    for c in cols:
        tasks.append((r, c))
tasks = list(set(tasks))  
if SEpoint not in tasks:
    tasks.append(SEpoint)
tasks.remove(SEpoint)
tasks = [SEpoint] + tasks

# ============================
# 2. 定义BFS搜索求解最短路径
# ============================

def bfs_path(grid, start, goal):
    """
    在grid上用BFS求从start到goal的最短路径。
    """
    rows, cols = grid.shape
    queue = deque([(start, [start])])
    visited = set([start])
    while queue:
        current, path = queue.popleft()
        if current == goal:
            return path
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                # 如果邻居为目标点，则允许进入；否则必须保证该点可通行（值为0）
                if (nr, nc) == goal or grid[nr, nc] == 0:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
    return None

def bfs_distance(grid, start, goal):
    """
    返回从start到goal的最短路径长度以及完整路径（如果无法到达，则返回无穷大距离）。
    """
    path = bfs_path(grid, start, goal)
    if path is None:
        return np.inf, None
    return len(path) - 1, path

# ============================
# 3. 构建任务点间的距离和路径矩阵
# ============================

n = len(tasks)
dist_matrix = [[None] * n for _ in range(n)]
path_matrix = [[None] * n for _ in range(n)]
for i in range(n):
    for j in range(n):
        if i != j:
            d, p = bfs_distance(cache, tasks[i], tasks[j])
            dist_matrix[i][j] = d
            path_matrix[i][j] = p
        else:
            dist_matrix[i][j] = 0
            path_matrix[i][j] = [tasks[i]]

# ============================
# 4. 利用贪心邻近搜索构造TSP路径
# ============================

# 从（索引0）开始
unvisited = set(range(1, n))  
route = [0]
current = 0
while unvisited:
    next_node = min(unvisited, key=lambda j: dist_matrix[current][j])
    route.append(next_node)
    unvisited.remove(next_node)
    current = next_node
route.append(0)

# ============================
# 5. 拼接各段路径并计算总路程
# ============================

full_path = []
total_length = 0
for i in range(len(route) - 1):
    segment = path_matrix[route[i]][route[i+1]]
    if segment is None:
        print(f"节点 {tasks[route[i]]} 到 {tasks[route[i+1]]} 不可达！")
        continue
    if i > 0:
        full_path.extend(segment[1:])
    else:
        full_path.extend(segment)
    total_length += dist_matrix[route[i]][route[i+1]]

print("任务访问顺序（索引）：", route)
print("任务访问顺序（坐标）：", [tasks[i] for i in route])
print("路径总长度：", total_length)
print("完整的路径步序：")
for step in full_path:
    print(step)

# ============================
# 6. 绘制仓库布局和路径轨迹
# ============================

plt.figure(figsize=(10, 10))
for r in range(cache.shape[0]):
    for c in range(cache.shape[1]):
        if cache[r, c] == 1:
            plt.scatter(c, -r, c='black', s=100)

path_x = [p[1] for p in full_path]
path_y = [-p[0] for p in full_path]
plt.plot(path_x, path_y, color='red', linewidth=2, marker='o', label='路径轨迹')

for i in range(len(full_path) - 1):
    start = full_path[i]
    end = full_path[i+1]
    plt.annotate("",
                 xy=(end[1], -end[0]), 
                 xytext=(start[1], -start[0]),
                 arrowprops=dict(arrowstyle="->", color="red", lw=1),
                 size=10)

task_x = [t[1] for t in tasks]
task_y = [-t[0] for t in tasks]
plt.scatter(task_x, task_y, color='blue', s=100, marker='s', label='任务点')

plt.scatter(SEpoint[1], -SEpoint[0], color='purple', s=150, marker='*', label='起始点')

plt.title(f"路径轨迹，总长度：{total_length}")
plt.xlabel("列")
plt.ylabel("行（倒置显示）")
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()
