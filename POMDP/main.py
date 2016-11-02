import envs.connectfour
import solvers.mcts
import numpy as np


max_time = 1
max_depth = 45
exploration = 2
board = np.zeros(shape=(6,7), dtype = np.int) # rows by columns
env = envs.connectfour.ConnectFour(start = board)
solver = solvers.mcts.MonteCarlo(env, max_time = max_time, max_depth = max_depth, exploration= exploration)

print board


print solver.to_tuple(solver.start)
