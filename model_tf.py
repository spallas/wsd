import os

import tensorflow as tf
import numpy as np
from data_preprocessing import load_dataset
from sklearn.metrics import f1_score, accuracy_score


def _attention(input_x, input_mask, W_att):
    h_masked = tf.boolean_mask(input_x, input_mask)
    h_tanh = tf.tanh(h_masked)
    u = tf.matmul(h_tanh, W_att)
    a = tf.nn.softmax(u)
    c = tf.reduce_sum(tf.multiply(h_tanh, a), 0)
    return c


LOG_DIR = "../log/"
SAVE_DIR = "../saved/"


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
        data = load_dataset()
        self.x_train, self.y_train, self.y_pos, self.x_mask, \
            self.sense_mask, self.embeddings, self.x_lengths = data["train"]
        self.dev_dict = data["dev_dict"]  # contains as keys the names of each corpus i.e.: senseval2, 3, etc.
        self.possible_senses = data["poss_dict"]
        # a dict: k = index of word, v = list of sense ids
        self.num_epochs = 3

        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        if not os.path.exists(LOG_DIR):
            os.mkdir(LOG_DIR)

    def build_graph_and_train(self, training=True):

        num_senses = 25913 + 1  # size of senses one hot repr # what is this???
        num_pos = 12
        max_sent_size = 258  # ???
        l2_lambda = 0.001
        kernel_size = 5
        num_filters = 128

        summaries = []

        # losses = [] ...for multiple GPUs..
        # predictions = []
        predictions_pos = []

        # =========== MODEL =========== #

        graph = tf.Graph()
        with graph.as_default():

            global_step = tf.Variable(0, trainable=False, name="global_step")
            learning_rate = tf.train.exponential_decay(self.start_learning_rate, global_step,
                                                       self.decay_steps, self.decay_rate, staircase=True)

            x = tf.placeholder('int32', [self.batch_size, max_sent_size], name="x")
            y = tf.placeholder('int32', [self.batch_size, max_sent_size], name="y")
            y_pos = tf.placeholder('int32', [self.batch_size, max_sent_size], name="y_pos")
            x_mask = tf.placeholder('bool', [self.batch_size, max_sent_size], name='x_mask')
            sense_mask = tf.placeholder('bool', [self.batch_size, max_sent_size], name='sense_mask')
            embeddings = tf.placeholder('float32', [self.vocab_size, self.embedding_size], name="embeddings")

            x_len = tf.reduce_sum(tf.cast(x_mask, 'int32'), 1)
            float_sense_mask = tf.cast(sense_mask, 'float')
            float_x_mask = tf.cast(x_mask, 'float')

            embeddings_output = tf.nn.embedding_lookup(embeddings, x)

            tile_x_mask = tf.tile(tf.expand_dims(float_x_mask, 2), [1, 1, self.embedding_size])
            Wx_masked = tf.multiply(embeddings_output, tile_x_mask)

            with tf.variable_scope("convolution"):
                conv1 = tf.layers.conv1d(inputs=Wx_masked, filters=num_filters, kernel_size=[kernel_size],
                                         padding='same', activation=tf.nn.relu, )
                conv2 = tf.layers.conv1d(inputs=conv1, filters=num_filters, kernel_size=[kernel_size], padding='same')

            # LSTM

            with tf.variable_scope("lstm1"):
                cell_fw1 = tf.contrib.rnn.LSTMCell(self.lstm_size)
                cell_bw1 = tf.contrib.rnn.LSTMCell(self.lstm_size)
                d_cell_fw1 = tf.contrib.rnn.DropoutWrapper(cell_fw1, input_keep_prob=self.input_dropout)
                d_cell_bw1 = tf.contrib.rnn.DropoutWrapper(cell_bw1, input_keep_prob=self.input_dropout)

                (output_fw1, output_bw1), state = tf.nn.bidirectional_dynamic_rnn(d_cell_fw1, d_cell_bw1, conv2,
                                                                                  sequence_length=x_len, dtype='float',
                                                                                  scope="lstm1")
                lstm_output1 = tf.concat([output_fw1, output_bw1], axis=-1)

            with tf.variable_scope("lstm2"):
                cell_fw2 = tf.contrib.rnn.LSTMCell(self.lstm_size)
                cell_bw2 = tf.contrib.rnn.LSTMCell(self.lstm_size)
                d_cell_fw2 = tf.contrib.rnn.DropoutWrapper(cell_fw2, input_keep_prob=self.input_dropout)
                d_cell_bw2 = tf.contrib.rnn.DropoutWrapper(cell_bw2, input_keep_prob=self.input_dropout)

                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(d_cell_fw2, d_cell_bw2, lstm_output1,
                                                                            sequence_length=x_len, dtype='float',
                                                                            scope="lstm2")
                lstm_output = tf.concat([output_fw, output_bw], axis=-1)

            # ATTENTION

            with tf.variable_scope("global_attention"):
                attention_mask = (tf.cast(x_mask, 'float') - 1) * 1e30
                W_att_global = tf.get_variable("W_att_global", shape=[2 * self.lstm_size, 1],
                                               initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                flat_h = tf.reshape(lstm_output, [self.batch_size * max_sent_size, tf.shape(lstm_output)[2]])
                h_tanh = tf.tanh(flat_h)
                u_flat = tf.matmul(h_tanh, W_att_global)
                u = tf.reshape(u_flat, [self.batch_size, max_sent_size]) + attention_mask
                a = tf.expand_dims(tf.nn.softmax(u, 1), 2)
                c = tf.reduce_sum(tf.multiply(lstm_output, a), axis=1)
                c_final = tf.tile(tf.expand_dims(c, 1), [1, max_sent_size, 1])
                h_final = tf.concat([c_final, lstm_output], 2)
                flat_h_final = tf.reshape(h_final, [self.batch_size * max_sent_size, tf.shape(h_final)[2]])

            with tf.variable_scope("hidden_layer"):
                W = tf.get_variable("W", shape=[4 * self.lstm_size, 2 * self.lstm_size],
                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                b = tf.get_variable("b", shape=[2 * self.lstm_size], initializer=tf.zeros_initializer())
                drop_flat_h_final = tf.nn.dropout(flat_h_final, self.input_dropout)
                flat_hl = tf.matmul(drop_flat_h_final, W) + b

            # OUTPUT

            with tf.variable_scope("softmax_layer"):
                W = tf.get_variable("W", shape=[2 * self.lstm_size, num_senses],
                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                b = tf.get_variable("b", shape=[num_senses], initializer=tf.zeros_initializer())
                drop_flat_hl = tf.nn.dropout(flat_hl, self.input_dropout)
                flat_logits_sense = tf.matmul(drop_flat_hl, W) + b
                logits = tf.reshape(flat_logits_sense, [self.batch_size, max_sent_size, num_senses])
                # predictions.append(tf.argmax(logits, 2))

            # Multi-task learning, train network to predict also part of speech
            with tf.variable_scope("softmax_layer_pos"):
                W = tf.get_variable("W", shape=[2 * self.lstm_size, num_pos],
                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                b = tf.get_variable("b", shape=[num_pos], initializer=tf.zeros_initializer())
                drop_flat_hl = tf.nn.dropout(flat_hl, self.input_dropout)
                flat_logits_pos = tf.matmul(drop_flat_hl, W) + b
                logits_pos = tf.reshape(flat_logits_pos, [self.batch_size, max_sent_size, num_pos])
                predictions_pos.append(tf.argmax(logits_pos, 2))

            loss = tf.contrib.seq2seq.sequence_loss(logits, y, float_sense_mask, name="loss")
            loss_pos = tf.contrib.seq2seq.sequence_loss(logits_pos, y_pos, float_x_mask, name="loss_pos")

            l2_loss = l2_lambda * tf.losses.get_regularization_loss()

            total_loss = loss + loss_pos + l2_loss

            optimizer = tf.train.AdamOptimizer(learning_rate)

            train_op = optimizer.minimize(total_loss, global_step=global_step)

            # SUMMARIES

            summaries.append(tf.summary.scalar("loss", loss))
            summaries.append(tf.summary.scalar("loss_pos", loss_pos))
            summaries.append(tf.summary.scalar("total_loss", total_loss))
            summaries.append(tf.summary.scalar("learning_rate", learning_rate))
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))
            summary = tf.summary.merge(summaries)
            saver = tf.train.Saver(tf.global_variables())

        # =========== TRAINING =========== #

        with tf.Session(graph=graph) as sess:

            sess.run(tf.global_variables_initializer())

            if training:

                num_batches = int(len(self.x_train) / self.batch_size)
                summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

                for epoch in range(self.num_epochs):
                    for b in range(num_batches):

                        start = b * self.batch_size
                        end = (b + 1) * self.batch_size

                        bx = self.x_train[start:end]
                        by = self.y_train[start:end]
                        by_pos = self.y_pos[start:end]
                        bx_mask = self.x_mask[start:end]
                        bs_mask = self.sense_mask[start:end]

                        _, step, _summary = sess.run([train_op, global_step, summary],
                                                     feed_dict={x: bx, y: by, x_mask: bx_mask, y_pos: by_pos,
                                                                sense_mask: bs_mask, embeddings: self.embeddings})
                        summary_writer.add_summary(_summary, step)

                        if b is not 0 and b % 100 == 0:
                            print()
                        else:
                            print("#", end="", flush=True)
                    saver.save(sess, save_path=SAVE_DIR)
                    print("Epoch => ", epoch + 1, ": Done")

            else:  # evaluating

                for corpus in self.dev_dict:

                    x_dev, y_dev, y_pos_dev, x_mask_dev, sense_mask_dev, sent_len = self.dev_dict[corpus]

                    num_batches = int(len(x_dev) / self.batch_size)

                    # saver.restore(sess, tf.train.latest_checkpoint(SAVE_DIR))

                    eval_predictions = []
                    truth = []
                    eval_predictions_pos = []
                    truth_pos = []

                    for b in range(num_batches):

                        start = b * self.batch_size
                        end = (b + 1) * self.batch_size

                        bx = x_dev[start:end]
                        by = y_dev[start:end]
                        by_pos = y_pos_dev[start:end]
                        bx_mask = x_mask_dev[start:end]
                        bs_mask = sense_mask_dev[start:end]
                        _loss, _logits, pred_pos = sess.run([total_loss, logits, predictions_pos],
                                                            feed_dict={x: bx, y: by, x_mask: bx_mask,
                                                                       y_pos: by_pos, sense_mask: bs_mask,
                                                                       embeddings: self.embeddings})
                        predictions = []
                        ground_truth = []

                        for i in range(self.batch_size):
                            for j in range(max_sent_size):
                                # for s in range(num_senses):
                                if bx_mask[i, j]:
                                    if bs_mask[i, j]:
                                        if bx[i, j] not in self.possible_senses:
                                            predictions.append(25913)
                                            # TODO: word never seen in training: append most frequent sense taken
                                            #       from external source
                                        else:
                                            possible_logits = []
                                            possible = list(self.possible_senses[bx[i, j]])
                                            for s in possible:
                                                possible_logits.append(_logits[i, j, s])
                                            predictions.append(possible[int(np.argmax(possible_logits))])
                                            # predictions.append(self.possible_senses[bx[i, j]].most_common(1)[0][0]).02
                                            # predictions.append(25913) 0.12
                                        ground_truth.append(int(by[i, j]))
                        eval_predictions += predictions
                        truth += ground_truth
                        # eval_predictions_pos += pred_pos

                    f1_sense = f1_score(truth, eval_predictions, average="micro")
                    acc_sense = accuracy_score(truth, eval_predictions)

                    # f1_pos = f1_score(self.y_pos_dev, eval_predictions_pos, average="micro")
                    # acc_pos = accuracy_score(self.y_pos_dev, eval_predictions_pos)

                    print("Evaluation on crorpus {}:".format(corpus))
                    print("\tF1 for senses => ", f1_sense, " - Accuracy senses => ", acc_sense)
                    # print("\tF1 for POS => ", f1_pos, " - Accuracy POS => ", acc_pos)

        return
