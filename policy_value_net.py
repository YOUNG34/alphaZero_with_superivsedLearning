import numpy as np
import tensorflow as tf
import chess
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

class Policy_value_net():

    def __init__(self,model_file = None):
        self.model = None

        self.input = tf.placeholder(tf.float32,shape=[None,18,8,8])
        self.input = tf.transpose(self.input,[0,2,3,1])

        # convoluiton layer one
        self.conv1 = tf.layers.conv2d(inputs=self.input,
                                      filters=256,kernal_size=3,
                                      padding="same",data_format="channels_last")
        self.conv1 = tf.layers.batch_normalization(inputs=self.conv1,axis=1)
        self.conv1 = tf.keras.layers.Activation("relu",name="input_relu")(self.conv1)

        # resudual block
        self.residual = self.conv1

        for i in range(19):
            self.residual = self.residual_block(self.residual,i+1)

        res_out = self.residual

        # for policy output
        self.conv2 = tf.layers.conv2d(inputs=res_out,
                                      filters=2, kernel_size=1,
                                      data_format="channels_first",
                                      kernel_regularizer=12*(1e-4))
        self.conv2 = tf.layers.batch_normalization(inputs=self.conv2,axis=1,name="batch_normalization")
        self.conv2 = tf.keras.layers.Activation("relu",name="policy_relu")(self.conv2)
        self.conv2 = tf.layers.flatten(self.conv2,name="policy_flatten")
        self.policy_out = tf.layers.dense(inputs=self.conv2,
                                     units=256,
                                     kernel_regularizer=12*(1e-4),
                                     activation=tf.nn.softmax,
                                     name="policy_out")

        # for value output
        self.conv3 = tf.layers.conv2d(inputs=res_out,
                                      filters=4,
                                      kernel_size=1,
                                      data_format="channels_first",
                                      use_bias=False,kernel_regularizer=12*(1e-4),
                                      name="value_conv-1-4")
        self.conv3 = tf.layers.batch_normalization(inputs=self.conv3,
                                                   axis=1,
                                                   name="value_batchnorm")
        self.conv3 = tf.keras.layers.Activation("relu",name="value_relu")(self.conv3)
        self.conv3 = tf.layers.flatten(inputs=self.conv3,name="value_flatten")
        self.conv3 = tf.layers.dense(inputs=self.conv3,
                                     units=256,
                                     kernel_regularizer=12*(1e-4),
                                     name="value_dense")
        self.conv3 = tf.layers.dense(inputs=self.conv3,
                                    units=1,
                                    kernel_regularizer=12*(1e-4),
                                    name="value_dense")
        self.value_out = tf.keras.layers.Activation("tanh",name="value_out")

        self.model = tf.keras.Model(self.input,[self.policy_out, self.value_out], name="chess_model")


        # define the loss function
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        # value loss function
        self.value_loss = tf.losses.mean_squared_error(self.labels,
                                                       self.value_out)
        # policy loss function
        self.mcts_probs = tf.placeholder(tf.float32, shape=[None,8*8])
        self.policy_loss = tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.multiply(self.mcts_probs, self.policy_out), 1)))
        # L2 penalty(regularization)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()]
        )
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        # Define the optimizer which is used for training
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)

        # Make a session
        self.session = tf.Session()

        # Policy entropy
        self.entropy = tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.exp(self.policy_out)*self.policy_out, 1)))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)

    def current_state(self):
        square_state = np.zeros((12, 8, 8))
        board = chess.BaseBoard()
        for i in range(6):
            s = board.pieces(i + 1, True)
            for white_loc in s:
                square_state[i][7 - white_loc // 8, white_loc % 8] = 1

        for black_i in range(6):
            black_s = board.pieces(black_i + 1, False)
            for black_loc in black_s:
                square_state[black_i + 6][7 - black_loc // 8, black_loc % 8] = 1
        return square_state

    def residual_block(self, input, index):

        res_name = "res" + index
        x = tf.layers.conv2d(inputs=input,
                             filters=256,
                             kernel_size=3,
                             padding="same",
                             data_format="channels_first",
                             use_bias=False,
                             kernel_regularize=12*(1e-4),
                             name=res_name+"_conv1-"+str(3)+"-"+str(256),
                             activation=tf.nn.relu
                             )
        x = tf.layers.BatchNormalization(axis=1, name=res_name + "_batchnorm1")(x)
        x = tf.keras.layers.Activation("relu", name=res_name + "_relu1")(x)
        x = tf.layers.Conv2D(filters=256, kernel_size=3, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=12*(1e-4),
                   name=res_name + "_conv2-" + str(3) + "-" + str(256))(x)
        x = tf.layers.BatchNormalization(axis=1, name="res" + str(index) + "_batchnorm2")(x)
        x = tf.keras.layers.Add(name=res_name + "_add")([self.input, x])
        x = tf.keras.layers.Activation("relu", name=res_name + "_relu2")(x)

        return x

    def policy_value(self, state_batch):
        log_act_probs,value = self.session.run(
            [self.policy_out,self.value_out],
            feed_dict={self.input:state_batch}
        )
        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def policy_value_fn(self,board):
        legal_positions = board.generate_legal_moves()
        current_state = np.ascontiguousarray(self.current_state().reshape(-1,12,8,8))
        act_probs, value = self.policy_value(current_state)
        act_probs = zip(legal_positions,act_probs[0][legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, entropy, _ = self.session.run(
            [self.loss, self.entropy, self.optimizer],
            feed_dict={self.input:state_batch,
                       self.mcts_probs:mcts_probs,
                       self.labels: winner_batch,
                       self.learning_rate: lr})
        return loss, entropy


    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self,model_path):
        self.saver.restore(self.session, model_path)










