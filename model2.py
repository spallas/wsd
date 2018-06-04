import os
import time

import tensorflow as tf
import numpy as np
from data_preprocessing import load_dataset, generate_batch
from sklearn.metrics import f1_score, accuracy_score


LOG_DIR = "../log2/"
SAVE_DIR = "../saved2/"


class Model2:

    def __init__(self, config):

        self.embedding_size = config["embed_size"]
        self.vocab_size = config["vocab_size"]
        self.lstm_size = config["lstm_size"]
        self.input_dropout = config["input_dropout"]
        self.batch_size = config["batch_size"]
        self.window_size = config["window_size"]
        self.start_learning_rate, \
            self.decay_steps, \
            self. decay_rate = config["learning_rate"]
        data = load_dataset()
        self.x_train, self.y_train, self.y_sen, self.y_pos, self.x_mask, \
            self.sense_mask, self.embeddings, self.x_lengths = data["train"]
        self.dev_dict = data["dev_dict"]  # contains as keys the names of each corpus i.e.: senseval2, 3, etc.
        self.possible_senses = data["poss_dict"]
        # a dict: k = index of word, v = list of sense ids
        self.num_epochs = config["num_epochs"]
        self.clipping = config["clip_grads"]

        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        if not os.path.exists(LOG_DIR):
            os.mkdir(LOG_DIR)

    def build_graph_and_train(self, training=True):

        num_senses = 25913 + 1  # size of senses one hot repr # what is this???
        num_pos = 12
        window_size = self.window_size
        l2_lambda = 0.001

        summaries = []

        # predictions = []
        predictions_pos = []

        # =========== MODEL =========== #

        graph = tf.Graph()
        with graph.as_default():

            global_step = tf.Variable(0, trainable=False, name="global_step")
            learning_rate = tf.train.exponential_decay(self.start_learning_rate, global_step,
                                                       self.decay_steps, self.decay_rate, staircase=True)

            x = tf.placeholder('int32', [self.batch_size, window_size], name="x")
            y = tf.placeholder('int32', [self.batch_size, window_size], name="y")
            y_sen = tf.placeholder('int32', [self.batch_size, window_size], name="y_sen")
            y_pos = tf.placeholder('int32', [self.batch_size, window_size], name="y_pos")
            x_mask = tf.placeholder('bool', [self.batch_size, window_size], name='x_mask')
            sense_mask = tf.placeholder('bool', [self.batch_size, window_size], name='sense_mask')
            embeddings = tf.placeholder('float32', [self.vocab_size, self.embedding_size], name="embeddings")

            x_len = tf.reduce_sum(tf.cast(x_mask, 'int32'), 1)
            float_sense_mask = tf.cast(sense_mask, 'float')
            float_x_mask = tf.cast(x_mask, 'float')

            embeddings_output = tf.nn.embedding_lookup(embeddings, x)

            # LSTM

            with tf.variable_scope("lstm1"):
                cell_fw1 = tf.contrib.rnn.LSTMCell(self.lstm_size)
                cell_bw1 = tf.contrib.rnn.LSTMCell(self.lstm_size)
                # d_cell_fw1 = tf.contrib.rnn.DropoutWrapper(cell_fw1, input_keep_prob=self.input_dropout)
                # d_cell_bw1 = tf.contrib.rnn.DropoutWrapper(cell_bw1, input_keep_prob=self.input_dropout)

                (output_fw1, output_bw1), state = tf.nn.bidirectional_dynamic_rnn(cell_fw1, cell_bw1, embeddings_output,
                                                                                  sequence_length=x_len, dtype='float',
                                                                                  scope="lstm1")
                lstm_output1 = tf.concat([output_fw1, output_bw1], axis=-1)

            with tf.variable_scope("lstm2"):
                cell_fw2 = tf.contrib.rnn.LSTMCell(self.lstm_size)
                cell_bw2 = tf.contrib.rnn.LSTMCell(self.lstm_size)
                # d_cell_fw2 = tf.contrib.rnn.DropoutWrapper(cell_fw2, input_keep_prob=self.input_dropout)
                # d_cell_bw2 = tf.contrib.rnn.DropoutWrapper(cell_bw2, input_keep_prob=self.input_dropout)

                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw2, cell_bw2, lstm_output1,
                                                                            sequence_length=x_len, dtype='float',
                                                                            scope="lstm2")
                lstm_output = tf.concat([output_fw, output_bw], axis=-1)
                flat_h = tf.reshape(lstm_output, [self.batch_size * window_size, tf.shape(lstm_output)[2]])

            # OUTPUT

            with tf.variable_scope("softmax_layer"):
                W = tf.get_variable("W", shape=[2 * self.lstm_size, self.vocab_size],
                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                b = tf.get_variable("b", shape=[self.vocab_size], initializer=tf.zeros_initializer())
                # drop_flat_hl = tf.nn.dropout(flat_hl, self.input_dropout)
                flat_logits = tf.matmul(flat_h, W) + b
                logits = tf.reshape(flat_logits, [self.batch_size, window_size, self.vocab_size])

            with tf.variable_scope("softmax_layer_sen"):
                W = tf.get_variable("W", shape=[2 * self.lstm_size, num_senses],
                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                b = tf.get_variable("b", shape=[num_senses], initializer=tf.zeros_initializer())
                # drop_flat_hl = tf.nn.dropout(flat_hl, self.input_dropout)
                flat_logits_sense = tf.matmul(flat_h, W) + b
                logits_sen = tf.reshape(flat_logits_sense, [self.batch_size, window_size, num_senses])

            # Multi-task learning, train network to predict also part of speech
            with tf.variable_scope("softmax_layer_pos"):
                W = tf.get_variable("W", shape=[2 * self.lstm_size, num_pos],
                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                b = tf.get_variable("b", shape=[num_pos], initializer=tf.zeros_initializer())
                # drop_flat_hl = tf.nn.dropout(flat_hl, self.input_dropout)
                flat_logits_pos = tf.matmul(flat_h, W) + b
                logits_pos = tf.reshape(flat_logits_pos, [self.batch_size, window_size, num_pos])
                predictions_pos.append(tf.argmax(logits_pos, 2))

            loss = tf.contrib.seq2seq.sequence_loss(logits, y, float_x_mask, name="loss")
            loss_sen = tf.contrib.seq2seq.sequence_loss(logits_sen, y_sen, float_sense_mask, name="loss_sen")
            loss_pos = tf.contrib.seq2seq.sequence_loss(logits_pos, y_pos, float_x_mask, name="loss_pos")

            l2_loss = l2_lambda * tf.losses.get_regularization_loss()

            total_loss = loss + loss_sen + loss_pos + l2_loss

            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            grads_vars = optimizer.compute_gradients(total_loss)

            clipped_grads = grads_vars
            if self.clipping:
                clipped_grads = [(tf.clip_by_norm(grad, 1), var) for grad, var in clipped_grads]

            train_op = optimizer.apply_gradients(clipped_grads, global_step=global_step)

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
                print("Starting to train...", flush=True)

                summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

                for epoch in range(self.num_epochs):
                    begin = time.time()
                    while True:
                        bx, by, by_sen, by_pos, bx_mask, bs_mask, reset = generate_batch(self.x_train, self.y_train,
                                                                                         self.y_sen, self.y_pos,
                                                                                         self.x_mask, self.sense_mask,
                                                                                         train=True)
                        if reset:
                            break
                        _, step, _summary = sess.run([train_op, global_step, summary],
                                                     feed_dict={x: bx, y: by, x_mask: bx_mask, y_pos: by_pos,
                                                                y_sen: by_sen, sense_mask: bs_mask,
                                                                embeddings: self.embeddings})
                        summary_writer.add_summary(_summary, step)

                        if step is not 0 and step % 50 == 0:
                            print(" Step: {}".format(step), flush=True)
                            print("Time for {} steps = {}".format(step, time.time() - begin))
                        print("#", end="", flush=True)

                    saver.save(sess, save_path=SAVE_DIR)
                    print("Epoch => ", epoch + 1, ": Done")

                    print("Evaluating performance...", flush=True)
                    for corpus in self.dev_dict:

                        x_dev, y_dev, y_sen_dev, y_pos_dev, x_mask_dev, sense_mask_dev, sent_len = self.dev_dict[corpus]

                        # saver.restore(sess, tf.train.latest_checkpoint(SAVE_DIR))

                        eval_predictions = []
                        truth = []

                        stop = False
                        while not stop:

                            bx, by, by_sen, by_pos, bx_mask, bs_mask, stop = generate_batch(x_dev, y_dev, y_sen_dev,
                                                                                            y_pos_dev, x_mask_dev,
                                                                                            sense_mask_dev, train=False)

                            _loss, _logits, pred_pos = sess.run([total_loss, logits_sen, predictions_pos],
                                                                feed_dict={x: bx, y: by, x_mask: bx_mask, y_pos: by_pos,
                                                                           y_sen: by_sen, sense_mask: bs_mask,
                                                                           embeddings: self.embeddings})
                            predictions = []
                            ground_truth = []

                            for i in range(self.batch_size):
                                for j in range(window_size):
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
                                            ground_truth.append(int(by_sen[i, j]))
                            eval_predictions += predictions
                            truth += ground_truth

                        f1_sense = f1_score(truth, eval_predictions, average="micro")
                        acc_sense = accuracy_score(truth, eval_predictions)

                        print("Corpus {}:\tF1 for senses => {}".format(corpus, f1_sense))
        return
