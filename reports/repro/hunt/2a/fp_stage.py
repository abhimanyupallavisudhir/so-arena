import numpy as np
from so_arena.analysis.optimization import GameTree, TreeNode, TreeLeaf, evaluate_tree, BestOfN, Tilted
from so_arena.games.normal_form import NormalFormGame

def stage_tree(A, B, V=None):
    m,k = A.shape
    nodes, leaves = {}, {}
    bkids=[]
    for i in range(m):
        leafs=[]
        for j in range(k):
            lid=f"L{i}{j}"
            leaves[lid]=TreeLeaf(id=lid, rewards={"a":float(A[i,j]),"b":float(B[i,j])}, values={"v":float((V if V is not None else A)[i,j])})
            leafs.append(lid)
        nid=f"B{i}"; nodes[nid]=TreeNode(id=nid,key="kB",role="b",group="g1",children=leafs); bkids.append(nid)
    nodes["root"]=TreeNode(id="root",key="kA",role="a",group="g1",children=bkids)
    return GameTree(item_id="x",mechanism="m",root="root",nodes=nodes,leaves=leaves,roles=["a","b"])

# Shapley game: FP known to cycle; product of averaged marginals is not an equilibrium
A=np.array([[0,1,0],[0,0,1],[1,0,0]],float)
B=np.array([[0,0,1],[1,0,0],[0,1,0]],float)
t=stage_tree(A,B)
for it in (200,201,1000,5000):
    tv=evaluate_tree(t,{"a":Tilted(float('inf')),"b":Tilted(float('inf'))},fp_iters=it)
    print("Shapley fp_iters",it,"rewards",{k:round(v,4) for k,v in tv.rewards.items()})
g=NormalFormGame(["a","b"],{"a":list("012"),"b":list("012")},{"a":A,"b":B})
print("unique NE payoff:", [round(g.expected(eq,player=p),4) for eq in g.support_enumeration() for p in "ab"])

# 2x2 game with unique mixed NE (matching pennies variant), exact BR
A=np.array([[3,0],[0,1]],float); B=-A
t=stage_tree(A,B)
tv=evaluate_tree(t,{"a":Tilted(float('inf')),"b":Tilted(float('inf'))})
print("zero-sum 2x2 (value 0.75):", tv.rewards)

print("--- perturbed Shapley game")
A=np.array([[0,1,0],[0,0,1],[1,0,0]],float)+np.array([[0.01,0,0],[0,0,0],[0,0,0]])
B=np.array([[0,0,1],[1,0,0],[0,1,0]],float)
t=stage_tree(A,B)
g=NormalFormGame(["a","b"],{"a":list("012"),"b":list("012")},{"a":A,"b":B})
print("NE:", [[np.round(x,3) for x in eq] for eq in g.support_enumeration()], "payoffs", [round(g.expected(eq,player=p),4) for eq in g.support_enumeration() for p in "ab"])
import so_arena.analysis.optimization as O
for it in (200,400,1000,3000):
    tv=evaluate_tree(t,{"a":Tilted(float('inf')),"b":Tilted(float('inf'))},fp_iters=it)
    print("fp_iters",it,"rewards",{k:round(v,4) for k,v in sorted(tv.rewards.items())})
