
import envs.discretegame
import envs.connectfour
import solvers.mcts
import numpy as np

# 7 columns by 6 rows, therefore 7 actions


max_time = 5
max_depth = 45
exploration = 2
board = np.zeros(shape=(6,7), dtype = np.int) # rows by columns
board[0,0] = 1
board[0,1] = 1
board[0,2] = 1

env = envs.connectfour.ConnectFour(start = board)
solver = solvers.mcts.MonteCarlo(env, max_time = max_time, max_depth = max_depth, exploration= exploration)


solver.search(env, env.start)


# seems to be working
# normal board biased against the ego player - what if opponent doesn't adversarily use the tree?
# print outputs for one whole 45 move rollout, investigate whether there really aren't any draws

'''
env.state, reward =env.generative_model(env.state, 0, 2)
env.state, reward =env.generative_model(env.state, 1, 2)
env.state[4,1] = 1
env.state[3,2] = 1
env.state[2,3] = 1
env.state[1,4] = 1
env.print_state(env.state)
print ''


next_states = env.legal_next_states(env.state, 2)


for state in next_states:
	env.print_state(state)
	print ''
#print solver.to_tuple(solver.start)
'''