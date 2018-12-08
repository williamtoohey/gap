import networkx as nx
import numpy as np
import math
import itertools
from pulp import *
from networkx.algorithms import bipartite, matching
from collections import defaultdict
from sympy.utilities.iterables import multiset_permutations

#parses the provided file, which is assumed to have the followed format:
#first line is number of GAP instances, then for each instance: a line with the number of agents/jobs, a line for each agent giving the cost of running each job,
#a line for each agent giving the time required to run each job, and a line giving the time bounds for each agent
def read_file(fileName):
	F = open(fileName, "r")
	numProblems = int(F.readline())
	costs = []; processes = []; limits = []
	for i in range (0, numProblems):
		agents, jobs = [int(x) for x in F.readline().split(" ", 2)]
		jobCosts = []; processTimes = []
		for j in range(0, agents):
			jobCosts.append([int(x) for x in F.readline().split(" ", jobs)])
		for j in range(0, agents):
			processTimes.append([int(x) for x in F.readline().split(" ", jobs)])
		timeLimits = [int(x) for x in F.readline().split(" ", agents)]
		costs.append(jobCosts)
		processes.append(processTimes)
		limits.append(timeLimits)
	F.close()
	return costs, processes, limits

#constructs the linear program representing the GAP problem for our current instance
#details about the constraints/objective function can be found on pages 280-281 in Williamson/Shmoys
def create_lp(instance):
	prob = LpProblem("Generalized assignment problem", LpMinimize)
	agents = len(costs[instance])
	jobs = len(costs[instance][0])
	x = LpVariable.dicts("x", ((y) for x in indices for y in x), 0, cat="Continuous")
	prob += lpSum(x[(str(i), str(j))] * costs[instance][i][j] for j in range(jobs) for i in range(agents))
	for j in range(jobs):
		prob += lpSum(x[str(i), str(j)] for i in range(agents)) == 1
	for i in range(agents):
		prob += lpSum(x[str(i), str(j)] * processes[instance][i][j] for j in range(jobs)) <= limits[instance][i]
	for i in range(agents):
		for j in range(jobs):
			if processes[instance][i][j] > limits[instance][i]:
				prob += x[str(i), str(j)] == 0
	return prob, x

#calculates the number of "bins" each agent runs jobs in, based on the optimal LP solution
def num_bins():
	bins = []
	for x in indices:
		sum = 0
		for y in x:
			sum += dictionary[y].varValue
		bins.append(math.ceil(sum))
	return bins

#creates (job time, job index) tuples for the given agent and sorts them in decreasing job time order
def construct_process_ordering(instance, agent):
	tuples = []
	for j in range(len(processes[instance][agent])):
		tuples.append((processes[instance][i][j], j))
	tuples.sort(key = lambda x: x[0], reverse=True)
	return tuples

#adds edges to G, mapping jobs to the bins the agent run them in in the optimal LP solution (if a job is run in two bins, we draw two edges
#from the job into the bins for the agent)
def construct_edges(instance, agent):
	ordering = construct_process_ordering(instance, agent)
	filled = 0; binNum = 0
	for x in ordering:
		value = dictionary[(str(agent), str(x[1]))].varValue
		if value > 0:
			filled += value
			G.add_edge(x[1], (agent, binNum), weight=costs[instance][agent][x[1]])
			if filled >= 1:
				filled -= 1
				binNum += 1
				if filled > 0:
					G.add_edge(x[1], (agent, binNum), weight=costs[instance][agent][x[1]])

#constructs the bipartite graph (one set represents the jobs, the other represents the bins that our agents run jobs in)
def construct_graph(instance):
	agents = len(costs[instance])
	jobs = len(costs[instance][0])
	G.add_nodes_from([(j) for j in range(jobs)], bipartite=0)
	bins = num_bins()
	for i in range(agents):
		G.add_nodes_from([(i, j) for j in range(bins[i])], bipartite=1)
	for i in range(agents):
		construct_edges(instance, i)

#constructs our solution by finding a minimum cost matching on the bipartite graph
#we update the cost of the edges to use the maximum matching algorithm to get a minimum matching instead
def construct_matching(instance):
	maxWeight = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[0][2]['weight']
	for e in G.edges():
		G[e[0]][e[1]]['weight'] = 2 * maxWeight - G[e[0]][e[1]]['weight']
	match = matching.max_weight_matching(G)
	assignments = []
	for x in match:
		if isinstance(x[0], tuple):
			assignments.append((x[0][0], x[1]))
		else:
			assignments.append((x[1][0], x[0]))
	return assignments

#gets the total cost of our algorithm's output and determines the largest ratio of agent run time to agent time constraint in our solution
def evaluate_sol(instance, sol):
	print("Instance:", instance)
	print("The algorithm's solution is:", sol)
	cost = 0
	for j in sol:
		cost += costs[instance][j[0]][j[1]]
	print("Our solution has cost", cost)
	dict = defaultdict(int)
	for agent, job in sol:
		dict[agent] += processes[instance][agent][job]
	worstRatio = 1.0; worstOffender = -1; timeWorstOffender = 0
	for agent, job in sol:
		ratio = dict[agent] / limits[instance][agent]
		if limits[instance][agent] < dict[agent] and ratio > worstRatio:
			worstRatio = ratio
			worstOffender = agent
			timeWorstOffender = dict[agent]
	if worstOffender == -1:
		print("All agents ran within their time bounds!\n")
	else:
		print("Worst time ratio was", worstRatio, "with agent", worstOffender, "running for", timeWorstOffender, "units with time limit", limits[instance][worstOffender], "\n")

#reads the provided file (in the same directory), and for each GAP instance: constructs a linear program, solves it, and if there is a feasible solution constructs a bipartite graph and minimal matching and analyzes the solution
costs, processes, limits = read_file("%s" % (sys.argv[1]))
for i in range(5):
#for i in range(len(costs)):
	indices = [[(str(y), str(x)) for x in range(len(costs[i][0]))] for y in range(len(costs[i]))]
	lp, dictionary = create_lp(i)
	lp.solve()
	if LpStatus[lp.status] == "Infeasible":
		print("Problem has no solution given constraints.")
	else:
		G = nx.Graph()
		construct_graph(i)
		sol = construct_matching(G)
		evaluate_sol(i, sol)