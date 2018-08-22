
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_for_train import MCTSPlayer
from policy_value_net import Policy_value_net
from collections import defaultdict, deque
from run import Run,Supervised_learning
import chess
import random
import numpy as np
import os
import json
import matplotlib.pyplot as plt


class Train():

    def __init__(self, init_model = None):
        self.board_width = 8
        self.board_height = 8
        self.readed_files_count = 0
        self.valid_files_num = 0

        self.run_game = Run()

        self.learn_rate = 1e-6
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.chess_mcts_playout = 4
        self.mcts_num = 10
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 384
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5
        self.kl_targ = 0.02
        self.check_freq = 10
        self.game_batch_num = 15000
        self.best_win_ratio = 0.0

        self.loss_list = []
        self.entropy_list = []

        if init_model:
            self.policy_value_net = Policy_value_net(model_file=init_model)
        else:
            self.policy_value_net = Policy_value_net()

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.chess_mcts_playout,
                                      is_selfplay=1)

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            self.result, play_data = self.run_game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            #print(np.array(play_data))
            #print('play_data[0][0]',play_data[0][0])
            self.episode_len = len(play_data)
            self.data_buffer.extend(play_data)
            #print(self.data_buffer)

    def policy_update(self):
       # assert self.data_buffer.reshape() == (None,18,8,8)
        mini_batch = self.data_buffer #random.sample(self.data_buffer, self.batch_size)
        #print(np.array(mini_batch).shape)
        state_batch = np.array([data[0] for data in mini_batch])

        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)

            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            #print('new_probs, new_v',new_probs, new_v)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            #print('kl',kl)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        #print('self.lr_multiplier',self.lr_multiplier)
        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print('explained_var_old',explained_var_old)
        print('explained_var_new',explained_var_new)
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        print('loss,entropy',loss,entropy)
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.chess_mcts_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.mcts_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.run_game.play_game(current_mcts_player, pure_mcts_player)
            print(winner)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.mcts_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}, result:{}".format(
                    i + 1, self.episode_len, self.result))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()

                # check the performance of the current model,
                # and save the model params

                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    win_ratio = self.policy_evaluate()
                    print('win_ratio',win_ratio)
                    self.policy_value_net.save_model('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.mcts_num < 5000):
                            self.mcts_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')

    def collect_supervised_learning_data(self, index):
        supervised_learning = Supervised_learning()

        self.result, play_data = supervised_learning.supervised_learning_run(index)
        if play_data != False:
            play_data = list(play_data)[:]

            self.episode_len = len(play_data)
            self.data_buffer.extend(play_data)

        # print(self.data_buffer)

    def supervised_learning(self):

        self.batch_num = 0
        for i in range(self.game_batch_num):
            #path = '/Users/zeyang/Desktop/alphaGoTest-master/move_json_files'
            path = '/home/k1758068/Desktop/alphaGoTest-master/move_json_files'
            files = os.listdir(path)

            #for j in range(self.readed_files_count + 1 ,self.readed_files_count+21):
            self.collect_supervised_learning_data(i+1)
            if self.result == False:
                #print("Invilid file.")
                continue
            else:
                self.valid_files_num += 1
                print("batch i:{}, episode_len:{}, result:{}".format(
                    self.valid_files_num, self.episode_len, self.result))

            self.readed_files_count += 10
            if len(self.data_buffer) > self.batch_size:
                loss, entropy = self.policy_update()
                self.file_name = '/home/k1758068/Desktop/alphaGoTest-master/loss_entropy.json'
                with open(self.file_name, "a") as loss_file:
                    json.dump(str([loss, entropy]), loss_file)
                self.loss_list.append(loss)
                self.entropy_list.append(entropy)
                self.data_buffer = []


            # check the performance of the current model,
            # and save the model params
            #
            if (i+1) % self.check_freq == 0:
                # x = []
                # for k in range(len(self.entropy_list)):
                #     x.append(k+1)
                # plt.figure()
                # fig, ax = plt.subplots(2)
                # ax[0].plot(x, self.loss_list, linestyle='-', color='red')
                # ax[1].plot(x, self.entropy_list, linestyle='--', color='green')
                #
                # plt.figure().savefig('my_loss_and_entropy_figure.png')
                #
                # plt.show()
                print("current self-play batch: {}".format(i + 1))
                #win_ratio = self.policy_evaluate()
                #print('win_ratio', win_ratio)
                self.policy_value_net.save_model('./current_policy.model')
                self.data_buffer = []
                # if win_ratio > self.best_win_ratio:
                #     print("New best policy!!!!!!!!")
                #     self.best_win_ratio = win_ratio
                #     # update the best_policy
                #     self.policy_value_net.save_model('./best_policy.model')
                #     if (self.best_win_ratio == 1.0 and
                #             self.mcts_num < 5000):
                #         self.mcts_num += 1000
                #         self.best_win_ratio = 0.0

if __name__ == '__main__':
    training = Train('./current_policy.model')
    training.supervised_learning()
    #training.run()
