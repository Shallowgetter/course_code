from gurobipy import Model, GRB, quicksum
import math

nodes = list(range(9))      
customers = nodes[1:]
demands = {
    0: 0,
    1: 9,
    2: 18,
    3: 8,
    4: 10,
    5: 12,
    6: 15,
    7: 16,
    8: 12
}
capacity = 80
num_vehicles = 4  

dist = [
    # 0    1   2   3   4   5   6   7   8
    [  0, 19, 21, 16, 18, 15, 19, 16, 18],  # from 0
    [19,  0,  2,  4,  3,  4,  5,  5,  7],   # from 1
    [21,  2,  0,  5,  4,  6,  5,  6,  7],   # from 2
    [16,  4,  5,  0,  2,  1,  4,  2,  4],   # from 3
    [18,  3,  4,  2,  0,  3,  2,  3,  4],   # from 4
    [15,  4,  6,  1,  3,  0,  4,  2,  5],   # from 5
    [19,  5,  5,  4,  2,  4,  0,  3,  2],   # from 6
    [16,  5,  6,  2,  3,  2,  3,  0,  3],   # from 7
    [18,  7,  7,  4,  4,  5,  2,  3,  0]    # from 8
]

m = Model("CVRP")

x = {}
for i in nodes:
    for j in nodes:
        if i != j:
            x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
m.update()

m.setObjective(quicksum(dist[i][j] * x[i, j] for i in nodes for j in nodes if i != j), GRB.MINIMIZE)

# 1. 每个客户必须被访问一次：入度和出度均为 1
for i in customers:
    m.addConstr(quicksum(x[j, i] for j in nodes if j != i) == 1, name=f"in_{i}")
    m.addConstr(quicksum(x[i, j] for j in nodes if j != i) == 1, name=f"out_{i}")

# 2. 车场（0）的出发和返回车辆数不超过可用车辆数
m.addConstr(quicksum(x[0, j] for j in nodes if j != 0) <= num_vehicles, name="depot_out")
m.addConstr(quicksum(x[i, 0] for i in nodes if i != 0) <= num_vehicles, name="depot_in")
m.update()

m.Params.LazyConstraints = 1


def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        vals = model.cbGetSolution(model._x)
        graph = {i: set() for i in nodes}
        for (i, j) in model._x.keys():
            if vals[i, j] > 0.5:
                graph[i].add(j)
                graph[j].add(i)

        visited = set()
        comps = []
        for i in nodes:
            if i not in visited:
                stack = [i]
                comp = set()
                while stack:
                    cur = stack.pop()
                    if cur not in visited:
                        visited.add(cur)
                        comp.add(cur)
                        stack.extend(graph[cur] - visited)
                comps.append(comp)

        for comp in comps:
            if 0 not in comp:

                demand_sum = sum(demands[i] for i in comp if i != 0)

                r = math.ceil(demand_sum / capacity)

                model.cbLazy(quicksum(model._x[i, j] for i in comp for j in comp if i != j) <= len(comp) - r)

m._x = x

m.optimize(subtourelim)

if m.status == GRB.OPTIMAL:
    sol = m.getAttr('x', x)
    routes = []
    for j in nodes:
        if j != 0 and sol[0, j] > 0.5:
            route = [0, j]
            current = j
            while True:
                next_node = None
                for k in nodes:
                    if k != current and sol[current, k] > 0.5:
                        next_node = k
                        break
                if next_node is None or next_node == 0:
                    route.append(0)
                    break
                else:
                    route.append(next_node)
                    current = next_node
            routes.append(route)
    
    for idx, route in enumerate(routes):
        print(f"车辆 {idx+1} 路径: {' -> '.join(map(str, route))}")
else:
    print("未找到最优解")
