import numpy as np
import tensorflow as tf
from run import Run
import chess


class Policy_value_net():

    def __init__(self,model_file = None):
        board_width = 8
        board_height = 8
        self.board_width = board_width
        self.board_height = board_height
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')  # 0.001    #5e-3    #0.05    #
        tf.summary.scalar('learning_rate', self.learning_rate)

        self.training = tf.placeholder(tf.bool, name='training')
        self.training = True

        # Define the tensorflow neural network
        # 1. Input:
        self.input_states = tf.placeholder(
            tf.float32, shape=[None, 18, board_height, board_width]) # 'channels_first'
        self.input = tf.transpose(self.input_states, [0, 2, 3, 1]) # 'channels_last'
        # 2. Common Networks Layers
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        self.conv1 = tf.layers.conv2d(inputs=self.input,
                                      filters=256, kernel_size=[3, 3],
                                      padding='SAME', data_format="channels_last",
                                      kernel_regularizer=regularizer)
        self.conv1 = tf.contrib.layers.batch_norm(self.conv1, center=False, epsilon=1e-5, fused=True,
                                                  is_training=self.training)

        # Residual_block
        self.residual_layer = self.conv1
        for i in range(19):
            self.residual_layer = self.residual_block(self.residual_layer,i)

        # Policy_head
        self.policy_head = tf.layers.conv2d(inputs=self.residual_layer, filters=2,
                                            kernel_size=[1, 1], padding='SAME',
                                            data_format="channels_last",
                                            kernel_regularizer=regularizer)
        self.policy_head = tf.contrib.layers.batch_norm(self.policy_head, center=False, epsilon=1e-5,fused=True,
                                                        is_training=self.training,activation_fn=tf.nn.relu)
        self.policy_head = tf.reshape(self.policy_head,[-1,8*8*2])
        self.policy_head = tf.contrib.layers.fully_connected(self.policy_head, 1968, activation_fn=None)


        # Value_head
        self.value_head = tf.layers.conv2d(self.residual_layer,filters=1,kernel_size=[1,1],
                                           padding='SAME', kernel_regularizer=regularizer)
        self.value_head = tf.contrib.layers.batch_norm(self.value_head, center=False, epsilon=1e-5, fused=True,
                                                       is_training=self.training, activation_fn=tf.nn.relu)
        self.value_head = tf.reshape(self.value_head, [-1, 8 * 8 * 1])
        self.value_head = tf.contrib.layers.fully_connected(self.value_head, 256,
                                                            activation_fn=tf.nn.relu)
        self.value_head = tf.contrib.layers.fully_connected(self.value_head, 1,
                                                            activation_fn=tf.nn.tanh)


        # Define loss function
        self.pi_ = tf.placeholder(tf.float32, [None, 1968], name='pi')
        self.policy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.pi_, logits=self.policy_head)
        self.policy_loss = tf.reduce_mean(self.policy_loss)

        self.z_ = tf.placeholder(tf.float32, [None, 1], name='z')
        self.value_loss = tf.losses.mean_squared_error(labels=self.z_, predictions=self.value_head)
        self.value_loss = tf.reduce_mean(self.value_loss)
        tf.summary.scalar('mse_loss', self.value_loss)

        regular_variables = tf.trainable_variables()
        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer, regular_variables)

        self.loss = self.value_loss + self.policy_loss + self.l2_loss
        tf.summary.scalar('loss', self.loss)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # Optimizer
        self.optimizer = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate, momentum=0.9, use_nesterov=True)

        # Accuracy
        correct_prediction = tf.equal(tf.argmax(self.policy_head, 1), tf.argmax(self.pi_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction, name='accuracy')
        tf.summary.scalar('move_accuracy', self.accuracy)


        self.sess = tf.Session()

        #         self.sess.run(tf.local_variables_initializer())
        #         self.sess.run(tf.initialize_all_variables())

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.exp(self.policy_head) * self.policy_head, 1)))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # For saving and restoring
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)



    def residual_block(self, input, index):
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        res_name = "res" + str(index)
        orig = tf.identity(input)
        residual_layer = tf.layers.conv2d(inputs=input,
                                          filters=256,
                                          kernel_size=3,
                                          padding='SAME',
                                          data_format='channels_last',
                                          use_bias=False,
                                          kernel_regularizer=regularizer,
                                          name=res_name+"_conv1-"+str(3)+"-"+str(256),
                                          activation=tf.nn.relu)
        residual_layer = tf.contrib.layers.batch_norm(residual_layer, center=False, epsilon=1e-5, fused=True,
                                             is_training=self.training, activation_fn=tf.nn.relu)

        residual_layer = tf.layers.conv2d(inputs=residual_layer,
                                          filters=256, kernel_size=3, padding='SAME',
                                          data_format='channels_last', use_bias=False,
                                          kernel_regularizer=regularizer,
                                          name=res_name + "_conv2-" + str(3) + "-" + str(256))
        residual_layer = tf.contrib.layers.batch_norm(inputs=residual_layer, center=False,
                                                      epsilon=1e-5, fused=True,
                                                      is_training=self.training)
        output = tf.nn.relu(tf.add(orig,residual_layer))

        return output

    def policy_value(self, state_batch):
        log_act_probs,value = self.sess.run(
            [self.policy_head,self.value_head],
            feed_dict={self.input_states:state_batch}
        )
        act_probs = np.exp(log_act_probs)
        print('value',value)
        return act_probs, value

    def policy_value_fn(self,board):
        run_game = Run()
        legal_positions = []
        for i in list(board.generate_legal_moves()):
            legal_positions.append(i.uci())
        current_state = np.ascontiguousarray(self.current_state().reshape(-1,18,8,8))
        act_probs, value = self.policy_value(current_state)
        act_probs = zip(legal_positions, act_probs[0])
        print('act_probs,value', list(act_probs), value)
        return act_probs, value

    def alg_to_coord(alg):
        rank = 8 - int(alg[1])  # 0-7
        file = ord(alg[0]) - ord('a')  # 0-7
        return rank, file

    def current_state(self):
        square_state = np.zeros((12, 8, 8))
        chess_board = chess.Board()
        foo = chess_board.fen().split(' ')

        en_passant = np.zeros((8, 8), dtype=np.float32)

        if foo[3] != '-':
            eps = self.alg_to_coord(foo[3])
            en_passant[eps[0]][eps[1]] = 1

        fifty_move_count = int(foo[4])
        fifty_move = np.full((8, 8), fifty_move_count, dtype=np.float32)

        castling = foo[2]
        auxiliary_planes = [np.full((8, 8), int('K' in castling), dtype=np.float32),
                            np.full((8, 8), int('Q' in castling), dtype=np.float32),
                            np.full((8, 8), int('k' in castling), dtype=np.float32),
                            np.full((8, 8), int('q' in castling), dtype=np.float32),
                            fifty_move,
                            en_passant]
        ret = np.array(auxiliary_planes, dtype=np.float32)
        assert ret.shape == (6, 8, 8)

        board = chess.BaseBoard()
        for i in range(6):
            s = board.pieces(i + 1, True)
            for white_loc in s:
                square_state[i][7 - white_loc // 8, white_loc % 8] = 1

        for black_i in range(6):
            black_s = board.pieces(black_i + 1, False)
            for black_loc in black_s:
                square_state[black_i + 6][7 - black_loc // 8, black_loc % 8] = 1

        square_state = np.array(square_state, dtype=np.float32)
        state = np.vstack([square_state,ret])
        return state

    def train_step(self, positions, probs, winners, learning_rate):
        winners = np.reshape(winners, (-1, 1))
        loss, entropy, _ = self.sess.run(
            [self.loss, self.entropy, self.optimizer],
            feed_dict={self.input:positions,
                       self.pi_:probs,
                       self.z_: winners,
                       self.learning_rate: learning_rate,
                       self.training: True})
        return loss, entropy


    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)

    def restore_model(self,model_path):
        self.saver.restore(self.sess, model_path)











