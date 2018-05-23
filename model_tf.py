import tensorflow as tf
from data_preprocessing import load_dataset


def _attention(input_x, input_mask, W_att):
    h_masked = tf.boolean_mask(input_x, input_mask)
    h_tanh = tf.tanh(h_masked)
    u = tf.matmul(h_tanh, W_att)
    a = tf.nn.softmax(u)
    c = tf.reduce_sum(tf.multiply(h_tanh, a), 0)
    return c


def _load_train_data():
    print("Model => Loading data...")
    print("Model => Done")
    return load_dataset()


def _load_dev_data():
    return None, None, None


class Model:

    def __init__(self, config):

        self.embedding_size = config["embed_size"]
        self.vocab_size = config["vocab_size"]
        self.lstm_size = config["lstm_size"]
        self.input_dropout = config["input_dropout"]
        self.batch_size = config["batch_size"]
        self.start_learning_rate, \
            self.decay_steps, \
            self. decay_rate = config["learning_rate"]
        self.x_train, self.y_train, self.y_pos, self.x_mask, \
            self.sense_mask, self.embeddings, self.x_lengths = _load_train_data()
        self.x_devel, self.y_devel, self.x_mask_dev = _load_dev_data()
        self.num_epochs = 10

    def build_graph_and_train(self):

        num_senses = 25913 + 1  # size of senses one hot repr # what is this???
        num_pos = 12
        max_sent_size = 258  # ???
        l2_lambda = 0.001

        # losses = [] ...for multiple GPUs..
        predictions = []
        predictions_pos = []

        # =========== MODEL =========== #

        graph = tf.Graph()
        with graph.as_default():
            # inputs and labels placeholders???
            x = tf.placeholder('int32', [self.batch_size, max_sent_size], name="x")
            y = tf.placeholder('int32', [self.batch_size, max_sent_size], name="y")
            y_pos = tf.placeholder('int32', [self.batch_size, max_sent_size], name="y_pos")
            x_mask = tf.placeholder('bool', [self.batch_size, max_sent_size], name='x_mask')
            sense_mask = tf.placeholder('bool', [self.batch_size, max_sent_size], name='sense_mask')
            embeddings = tf.placeholder('float32', [self.vocab_size, self.embedding_size], name="embeddings")

            x_len = tf.reduce_sum(tf.cast(x_mask, 'int32'), 1)

            global_step = tf.Variable(0, trainable=False, name="global_step")
            learning_rate = tf.train.exponential_decay(self.start_learning_rate, global_step,
                                                       self.decay_steps, self.decay_rate, staircase=True)
            # Load embeddings???
            embeddings_output = tf.nn.embedding_lookup(embeddings, x)

            # LSTM

            cell_fw = tf.contrib.rnn.LSTMCell(self.lstm_size)
            cell_bw = tf.contrib.rnn.LSTMCell(self.lstm_size)

            d_cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=self.input_dropout)
            d_cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=self.input_dropout)

            (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(
                d_cell_fw, d_cell_bw, embeddings_output, sequence_length=x_len, dtype=tf.float32)

            lstm_output = tf.concat([output_fw, output_bw], axis=-1)

            # ATTENTION

            W_att = tf.Variable(tf.truncated_normal([2 * self.lstm_size, 1], mean=0.0, stddev=0.1, seed=0), name="W_att")
            c = tf.expand_dims(_attention(lstm_output[0], x_mask[0], W_att), 0)
            for i in range(1, self.batch_size):
                c = tf.concat([c, tf.expand_dims(_attention(lstm_output[i], x_mask[i], W_att), 0)], 0)

            cc = tf.expand_dims(c, 1)
            c_final = tf.tile(cc, [1, max_sent_size, 1])
            h_final = tf.concat([c_final, lstm_output], 2)
            flat_h_final = tf.reshape(h_final, [-1, 4 * self.lstm_size])

            # OUTPUT

            with tf.variable_scope("hidden_layer"):
                W = tf.get_variable("W", shape=[4 * self.lstm_size, 2 * self.lstm_size],
                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                b = tf.get_variable("b", shape=[2 * self.lstm_size], initializer=tf.zeros_initializer())
                drop_flat_h_final = tf.nn.dropout(flat_h_final, self.input_dropout)
                flat_hl = tf.matmul(drop_flat_h_final, W) + b

            with tf.variable_scope("softmax_layer"):
                W = tf.get_variable("W", shape=[2 * self.lstm_size, num_senses],
                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                b = tf.get_variable("b", shape=[num_senses], initializer=tf.zeros_initializer())
                drop_flat_hl = tf.nn.dropout(flat_hl, self.input_dropout)
                flat_logits_sense = tf.matmul(drop_flat_hl, W) + b
                logits = tf.reshape(flat_logits_sense, [self.batch_size, max_sent_size, num_senses])
                predictions.append(tf.argmax(logits, 2))

            # Multi-task learning, train network to predict also part of speech
            with tf.variable_scope("softmax_layer_pos"):
                W = tf.get_variable("W", shape=[2 * self.lstm_size, num_pos],
                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                b = tf.get_variable("b", shape=[num_pos], initializer=tf.zeros_initializer())
                drop_flat_hl = tf.nn.dropout(flat_hl, self.input_dropout)
                flat_logits_pos = tf.matmul(drop_flat_hl, W) + b
                logits_pos = tf.reshape(flat_logits_pos, [self.batch_size, max_sent_size, num_pos])
                predictions_pos.append(tf.arg_max(logits_pos, 2))

            float_sense_mask = tf.cast(sense_mask, 'float')
            float_x_mask = tf.cast(x_mask, 'float')

            loss = tf.contrib.seq2seq.sequence_loss(logits, y, float_sense_mask, name="loss")
            loss_pos = tf.contrib.seq2seq.sequence_loss(logits_pos, y_pos, float_x_mask, name="loss_pos")

            l2_loss = l2_lambda * tf.losses.get_regularization_loss()

            total_loss = loss + loss_pos + l2_loss

            optimizer = tf.train.AdamOptimizer(learning_rate)

            train_op = optimizer.minimize(total_loss, global_step=global_step)

        # =========== TRAINING =========== #

        num_batches = int(len(self.x_train) / self.batch_size)

        with tf.Session(graph=graph) as sess:

            sess.run(tf.global_variables_initializer())

            for epoch in range(self.num_epochs):

                for i in range(num_batches):

                    start = i * self.batch_size
                    end = (i+1) * self.batch_size

                    bx = self.x_train[start:end]
                    by = self.y_train[start:end]
                    by_pos = self.y_pos[start:end]
                    bx_mask = self.x_mask[start:end]
                    bs_mask = self.sense_mask[start:end]

                    sess.run(train_op, feed_dict={x: bx, y: by, x_mask: bx_mask, y_pos: by_pos,
                                                  sense_mask: bs_mask, embeddings: self.embeddings})
                    if i is 0:
                        print("Done first batch")
        return

    def evaluate(self):
        pass
