from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_for_train import MCTSPlayer
from policy_value_net import Policy_value_net
from collections import defaultdict, deque


class Train():

    def __init__(self, init_model = None):
        self.board_width = 8
        self.board_height = 8

        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 384
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 1000

        if init_model:
            self.policy_value_net = Policy_value_net(model_file=init_model)
        else:
            self.policy_value_net = Policy_value_net()

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
