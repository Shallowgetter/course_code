import pulp


Regions = ["NE", "ESE", "MCE", "S", "W"]   # 五大区域
ExistingFactories = ["A", "B"]            # 已有工厂
CandidateFactories = ["C", "D"]           # 候选工厂（只建其中一个）
AllFactories = ExistingFactories + CandidateFactories

# 1) 区域需求(万台)
demands = {
    "NE":  700,
    "ESE": 600,
    "MCE": 400,
    "S":   800,
    "W":   500
}

# 2) 工厂产能(万台)
capacity = {
    "A": 1000,
    "B": 1500,
    "C": 1500,
    "D": 1500
}

# 3) 固定成本(百万元) --> 转换成 "万元"
fixed_cost_raw = {
    "A": 100,  # 已有厂
    "B": 150,  # 已有厂
    "C": 170,  # 候选厂
    "D": 150   # 候选厂
}
fixed_cost = {}
for f in AllFactories:
    fixed_cost[f] = fixed_cost_raw[f] * 10000  

# 4) 单位生产+运输成本(元/台)。
variable_cost = {
    ("A","NE"):5, ("A","ESE"):6, ("A","MCE"):7, ("A","S"):8, ("A","W"):9,
    ("B","NE"):6, ("B","ESE"):5, ("B","MCE"):5, ("B","S"):7, ("B","W"):8,
    ("C","NE"):6, ("C","ESE"):6, ("C","MCE"):6, ("C","S"):7, ("C","W"):7,
    ("D","NE"):5, ("D","ESE"):5, ("D","MCE"):6, ("D","S"):6, ("D","W"):7
}

model = pulp.LpProblem("PC_Facility_Location", sense=pulp.LpMinimize)

# x_{f,r} >= 0: 从工厂f到区域r的发运量(万台)
x = pulp.LpVariable.dicts("Ship", 
                          ((f,r) for f in AllFactories for r in Regions),
                          lowBound=0, cat=pulp.LpContinuous)

# y_C, y_D in {0,1}: 是否选址在C或D
y = pulp.LpVariable.dicts("Open",
                          (f for f in CandidateFactories),
                          lowBound=0, upBound=1, cat=pulp.LpBinary)

model += (
    fixed_cost["A"] + fixed_cost["B"]  
    + fixed_cost["C"] * y["C"] + fixed_cost["D"] * y["D"] 
    + pulp.lpSum(
        variable_cost[(f,r)] * x[(f,r)]  
        for f in AllFactories
        for r in Regions
    )
)

for r in Regions:
    model += pulp.lpSum( x[(f,r)] for f in AllFactories ) == demands[r], f"Demand_{r}"

#     A,B必然开放(产能约束); C,D需乘以 y[C] 或 y[D]
model += pulp.lpSum( x[("A",r)] for r in Regions ) <= capacity["A"], "Cap_A"
model += pulp.lpSum( x[("B",r)] for r in Regions ) <= capacity["B"], "Cap_B"

model += pulp.lpSum( x[("C",r)] for r in Regions ) <= capacity["C"] * y["C"], "Cap_C"
model += pulp.lpSum( x[("D",r)] for r in Regions ) <= capacity["D"] * y["D"], "Cap_D"

model += y["C"] + y["D"] == 1, "Select_One_New_Fac"

model.solve(pulp.PULP_CBC_CMD(msg=0))  

print("Status:", pulp.LpStatus[model.status])

print("决策变量 y_C,y_D：")
for f in CandidateFactories:
    print(f"  y_{f} =", pulp.value(y[f]))

print("\n各工厂到各区域的最优发运量(万台)：")
for f in AllFactories:
    for r in Regions:
        ship_val = pulp.value(x[(f,r)])
        if ship_val > 1e-6: 
            print(f"  {f} -> {r}: {ship_val:.1f} 万台")

print("\n最小化后的总成本(万元) =", pulp.value(model.objective))
