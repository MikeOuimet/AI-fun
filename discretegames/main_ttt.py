import envs.tictactoe
import solvers.mcts


max_time = 1
max_depth = 10
exploration = 2
board = [0,0,0,0,0,0,0,0,0]
env = envs.tictactoe.TicTacToe(start = board)
solver = solvers.mcts.MonteCarlo(env, max_time = max_time, max_depth = max_depth, exploration= exploration)

env.play_game_MCTS(solver)

