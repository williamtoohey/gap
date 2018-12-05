import networkx as nx
import numpy as np
from pulp import *
from networkx.algorithms import bipartite

#B = nx.Graph()
#B.add_nodes_from([1,2,3,4], bipartite=0)
#B.add_nodes_from(['a','b','c'], bipartite=1)
#B.add_edges_from([(1,'a'), (1, 'b'), (2, 'b'), (2, 'c'), (3, 'c'), (4, 'a')])

#print(nx.is_connected(B))

#bottom, top = bipartite.sets(B)
#print(bottom)
#print(top)

#parses the provided file, which is assumed to have the followed format:
#first line is number of GAP instances, then for each instance: a line with the number of agents/jobs, a line for each agent giving the cost of running each job,
#a line for each agent giving the time required to run each job, a line giving the time bounds for each agent, and finally the optimal solution cost
def read_file(fileName):
	F = open("gap1.txt", "r")
	numProblems = int(F.readline())
	costs = []
	processes = []
	limits = []
	opts = []
	for i in range (0, numProblems):
		agents, jobs = [int(x) for x in F.readline().split(" ", 2)]
		jobCosts = []
		processTimes = []
		for j in range(0, agents):
			jobCosts.append([int(x) for x in F.readline().split(" ", jobs)])
		for j in range(0, agents):
			processTimes.append([int(x) for x in F.readline().split(" ", jobs)])
		timeLimits = [int(x) for x in F.readline().split(" ", agents)]
		costs.append(jobCosts)
		processes.append(processTimes)
		limits.append(timeLimits)
		opts.append(int(F.readline()))
	return costs, processes, limits, opts

#constructs the linear program representing the GAP problem for our current instance
#details about the constraints/object function can be found on pages 280-281 in Williamson/Shmoys
def create_lp(instance, costs, processes, limits):
	prob = LpProblem("Generalized assignment problem", LpMinimize)
	agents = len(costs[0])
	jobs = len(costs[0][0])
	indices = [[(str(x), str(y)) for x in range(agents)] for y in range(jobs)]
	x = LpVariable.dicts("x", ((y) for x in indices for y in x), 0, cat="Continuous")
	prob += lpSum(x[(str(i), str(j))] * costs[instance][i][j] for i in range(agents) for j in range(jobs))
	for j in range(jobs):
		prob += lpSum(x[str(i), str(j)] for i in range(agents)) == 1
	for i in range(agents):
		prob += lpSum(x[str(i), str(j)] * processes[instance][i][j] for j in range(jobs)) <= limits[instance][i]
	return prob

costs, processes, limits, opts = read_file("gap1.txt")
lp = create_lp(0, costs, processes, limits)
print(lp)
lp.solve()
for i in lp.variables():
	print(i.name, "=", i.varValue)