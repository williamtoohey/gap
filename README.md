# Generalized Assignment Problem

This was written and tested using Python 3.7.

First install pulp and networkx (through pip). As a command line argument, this code takes in the name of a text file that provides instances of the Generalized Assignment Problem. Several example files are included in the codebase.

Input text files are assumed to have the following format, which is a line for the number of GAP instances, and for each instance: a line giving the number of agents and jobs for the instance, then a line for each agent specifying the cost of running each job, then a line for each agent specifying the time required to run each job to completion, and then a line giving the time bounds for each agent.

The output is, for each instance provided, the solution produced by the algorithm detailed in section 11.1, the solution's cost, and the largest ratio of a machine's runtime compared to its time bound (which was shown to be at most 2 in the book).
