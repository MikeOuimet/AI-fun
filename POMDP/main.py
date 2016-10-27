import envs.tictactoe
import solvers.mcts

max_time = .1
max_depth = 10
env = envs.tictactoe.TicTacToe(start =[0,0,0,0,0,0,0,0,0])
solver = solvers.mcts.MonteCarlo(env, max_time = max_time, max_depth = max_depth)
#solver.search(solver.history)
#solver.rollout_policy = env.rollout_policy
#print solver.rollout(env, solver.state, solver.history, 0)
board = [0,0,0,0,2,0,0,0,0]
#board =[0, 1, 2, 0, 0, 0, 0, 0, 0]

#print env.two_ply_generative(board, 1)

#print len(env.legal_actions(board))


print solver.simulate(env, board, 0)


print 'The number of times visited are'
print solver.visits
print ''
print ' The current value estimates are'
print solver.values

