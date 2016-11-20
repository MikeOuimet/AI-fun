
import envs.discretegame
import envs.connectfour
import solvers.mcts
import numpy as np

#  6 rows by 7 columns, therefore 7 actions


max_time = 5
max_depth = 23
exploration = 2
human_first = False
board = np.zeros(shape=(6,7), dtype = np.int) # rows by columns
verbose = False
warm_start = False
#board[5,6] = 0
#board[2,1] = 2
#board[1,2] = 2
#board[0,3] = 2

env = envs.connectfour.ConnectFour(start = board, human_first= human_first)
solver = solvers.mcts.MonteCarlo(env, max_time = max_time, max_depth = max_depth, exploration= exploration, \
	verbose= verbose, warm_start = warm_start)

env.play_game_MCTS(solver)




'''
[0 0 0 0 0 0 0]
[0 0 0 0 0 0 0]
[1 0 0 0 0 0 0]
[2 1 0 2 0 0 0]
[2 1 1 1 2 0 0]
[2 2 1 1 1 2 0]
value -1.0
times visited 731
upper confidence bound -0.783700472051
'''

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