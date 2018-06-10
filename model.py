import os
import time

import tensorflow as tf
import numpy as np
from data_preprocessing import load_dataset, generate_batch, get_params_dict, load_test_set
from sklearn.metrics import f1_score, accuracy_score


LOG_DIR = "../log/"
SAVE_DIR = "..//saved"


class Model:

    def __init__(self):

        data = load_dataset()
        config = get_params_dict()

        self.embedding_size = config["embed_size"]
        self.vocab_size = config["vocab_size"]
        self.lstm_size = config["lstm_size"]
        self.input_dropout = config["input_dropout"]
        self.batch_size = config["batch_size"]
        self.window_size = config["window_size"]
        self.start_learning_rate, \
            self.decay_steps, \
            self. decay_rate = config["learning_rate"]
        self.x_train, self.y_train, self.y_sen, self.y_pos, self.x_mask, \
            self.sense_mask, self.embeddings, self.x_lengths = data["train"]
        self.dev_dict = data["dev_dict"]  # contains as keys the names of each corpus i.e.: senseval2, 3, etc.
        self.possible_senses = data["poss_dict"]
        # a dict: k = index of word, v = list of sense ids
        self.num_epochs = config["num_epochs"]
        self.clipping = config["clip_grads"]
        self.use_cnn, self.use_dropout, self.use_att = config["model"]
        self.sense2id, self.id2sense, self.id2pos, self.pos2id, self.word2id, self.MFS = data["maps"]
        self.id2word = {v: k for k, v in self.word2id.items()}
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        if not os.path.exists(LOG_DIR):
            os.mkdir(LOG_DIR)

    def build_graph_and_train(self, training=True):
        """
        Build the model graph and define the training, evaluation, and prediction
        steps.
        :param training: Whether to train or predict senses
        """
        num_senses = len(self.sense2id)
        num_pos = len(self.pos2id)
        window_size = self.window_size
        l2_lambda = 0.001
        # conv nets parameters
        kernel_size = 5
        num_filters = 128

        summaries = []
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
            sense_mask_f32 = tf.cast(sense_mask, 'float32')
            x_mask_f32 = tf.cast(x_mask, 'float32')

            embeddings_output = tf.nn.embedding_lookup(embeddings, x)
            lstm_input = embeddings_output

            tile_x_mask = tf.tile(tf.expand_dims(x_mask_f32, 2), [1, 1, self.embedding_size])
            Wx_masked = tf.multiply(embeddings_output, tile_x_mask)

            if self.use_cnn:
                with tf.variable_scope("convolution"):
                    conv1 = tf.layers.conv1d(inputs=Wx_masked, filters=num_filters, kernel_size=[kernel_size],
                                             padding='same', activation=tf.nn.relu)
                    conv2 = tf.layers.conv1d(inputs=conv1, filters=num_filters, kernel_size=[kernel_size], padding='same')
                    lstm_input = conv2

            # LSTM

            with tf.variable_scope("lstm1"):
                cell_fw1 = tf.contrib.rnn.LSTMCell(self.lstm_size)
                cell_bw1 = tf.contrib.rnn.LSTMCell(self.lstm_size)
                d_cell_fw1 = tf.contrib.rnn.DropoutWrapper(cell_fw1, input_keep_prob=self.input_dropout)
                d_cell_bw1 = tf.contrib.rnn.DropoutWrapper(cell_bw1, input_keep_prob=self.input_dropout)
                if self.use_dropout:
                    cell_fw1, cell_bw1 = d_cell_fw1, d_cell_bw1
                (output_fw1, output_bw1), state = tf.nn.bidirectional_dynamic_rnn(cell_fw1, cell_bw1, lstm_input,
                                                                                  sequence_length=x_len, dtype='float',
                                                                                  scope="lstm1")
                lstm_output1 = tf.concat([output_fw1, output_bw1], axis=-1)

            with tf.variable_scope("lstm2"):
                cell_fw2 = tf.contrib.rnn.LSTMCell(self.lstm_size)
                cell_bw2 = tf.contrib.rnn.LSTMCell(self.lstm_size)
                d_cell_fw2 = tf.contrib.rnn.DropoutWrapper(cell_fw2, input_keep_prob=self.input_dropout)
                d_cell_bw2 = tf.contrib.rnn.DropoutWrapper(cell_bw2, input_keep_prob=self.input_dropout)
                if self.use_dropout:
                    cell_fw2, cell_bw2 = d_cell_fw2, d_cell_bw2
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw2, cell_bw2, lstm_output1,
                                                                            sequence_length=x_len, dtype='float',
                                                                            scope="lstm2")
                lstm_output = tf.concat([output_fw, output_bw], axis=-1)

            flat_layer = tf.reshape(lstm_output, [self.batch_size * window_size, tf.shape(lstm_output)[2]])

            if self.use_att:
                with tf.variable_scope("attention"):
                    attention_mask = (tf.cast(x_mask, 'float') - 1) * 1e30
                    W_att = tf.get_variable("W_att", shape=[2 * self.lstm_size, 1],
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                    flat_h = tf.reshape(lstm_output, [self.batch_size * window_size, tf.shape(lstm_output)[2]])
                    h_tanh = tf.tanh(flat_h)
                    u_flat = tf.matmul(h_tanh, W_att)
                    u = tf.reshape(u_flat, [self.batch_size, window_size]) + attention_mask
                    a = tf.expand_dims(tf.nn.softmax(u, 1), 2)
                    c = tf.reduce_sum(tf.multiply(lstm_output, a), axis=1)
                    c = tf.tile(tf.expand_dims(c, 1), [1, window_size, 1])
                    att_output = tf.concat([c, lstm_output], 2)
                    flat_h_final = tf.reshape(att_output, [self.batch_size * window_size, tf.shape(att_output)[2]])

                with tf.variable_scope("hidden_layer"):
                    W = tf.get_variable("W", shape=[4 * self.lstm_size, 2 * self.lstm_size],
                                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                    b = tf.get_variable("b", shape=[2 * self.lstm_size], initializer=tf.zeros_initializer())
                    drop_flat_h_final = tf.nn.dropout(flat_h_final, self.input_dropout)
                    flat_layer = tf.matmul(drop_flat_h_final, W) + b

            # OUTPUT

            with tf.variable_scope("softmax_layer"):
                W = tf.get_variable("W", shape=[2 * self.lstm_size, self.vocab_size],
                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                b = tf.get_variable("b", shape=[self.vocab_size], initializer=tf.zeros_initializer())
                drop_flat_hl = tf.nn.dropout(flat_layer, self.input_dropout)
                if self.use_dropout:
                    flat_layer = drop_flat_hl
                flat_logits = tf.matmul(flat_layer, W) + b
                logits = tf.reshape(flat_logits, [self.batch_size, window_size, self.vocab_size])

            with tf.variable_scope("softmax_layer_sen"):
                W = tf.get_variable("W", shape=[2 * self.lstm_size, num_senses],
                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                b = tf.get_variable("b", shape=[num_senses], initializer=tf.zeros_initializer())
                drop_flat_hl = tf.nn.dropout(flat_layer, self.input_dropout)
                if self.use_dropout:
                    flat_layer = drop_flat_hl
                flat_logits_sense = tf.matmul(flat_layer, W) + b
                logits_sen = tf.reshape(flat_logits_sense, [self.batch_size, window_size, num_senses])

            # Multi-task learning, train network to predict also part of speech
            with tf.variable_scope("softmax_layer_pos"):
                W = tf.get_variable("W", shape=[2 * self.lstm_size, num_pos],
                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                b = tf.get_variable("b", shape=[num_pos], initializer=tf.zeros_initializer())
                drop_flat_hl = tf.nn.dropout(flat_layer, self.input_dropout)
                if self.use_dropout:
                    flat_layer = drop_flat_hl
                flat_logits_pos = tf.matmul(flat_layer, W) + b
                logits_pos = tf.reshape(flat_logits_pos, [self.batch_size, window_size, num_pos])
                predictions_pos.append(tf.argmax(logits_pos, 2))

            loss = tf.contrib.seq2seq.sequence_loss(logits, y, x_mask_f32, name="loss")
            loss_sen = tf.contrib.seq2seq.sequence_loss(logits_sen, y_sen, sense_mask_f32, name="loss_sen")
            loss_pos = tf.contrib.seq2seq.sequence_loss(logits_pos, y_pos, x_mask_f32, name="loss_pos")
            l2_loss = l2_lambda * tf.losses.get_regularization_loss()
            total_loss = loss + loss_sen + loss_pos + l2_loss

            optimizer = tf.train.AdamOptimizer(learning_rate)

            gradients = optimizer.compute_gradients(total_loss)

            clipped_grads = gradients
            if self.clipping:
                clipped_grads = [(tf.clip_by_norm(grad, 1), var) for grad, var in clipped_grads]

            train_op = optimizer.apply_gradients(clipped_grads, global_step=global_step)

            # SUMMARIES

            summaries.append(tf.summary.scalar("loss_sen", loss_sen))
            summaries.append(tf.summary.scalar("loss_pos", loss_pos))
            summaries.append(tf.summary.scalar("total_loss", total_loss))
            summaries.append(tf.summary.scalar("learning_rate", learning_rate))
            summary = tf.summary.merge(summaries)
            saver = tf.train.Saver(save_relative_paths=True)

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

                    # =========== EVALUATING =========== #

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
                                                sense = self.sense2id.get(self.MFS.get(bx[i, j], "bn:00000000x"), 25913)
                                                predictions.append(sense)
                                            else:
                                                possible_logits = []
                                                possible = self.filter_senses(bx[i, j], by_pos[i, j])
                                                for s in possible:
                                                    possible_logits.append(_logits[i, j, s])
                                                predictions.append(possible[int(np.argmax(possible_logits))])
                                            ground_truth.append(int(by_sen[i, j]))
                            eval_predictions += predictions
                            truth += ground_truth

                        f1_sense = f1_score(truth, eval_predictions, average="micro")
                        acc_sense = accuracy_score(truth, eval_predictions)

                        print("Corpus {}:\tF1 for senses => {}".format(corpus, f1_sense))
            else:

                # =========== TESTING =========== #

                # saver.restore(sess, tf.train.latest_checkpoint(SAVE_DIR))

                x_test, x_ids_test, x_mask_test, y_pos_test, sense_mask_test = load_test_set(self.pos2id, self.word2id)

                stop = False

                with open("answers.txt", "w") as f:
                    while not stop:
                        bx, bx_ids, by_pos, bx_mask, bs_mask, stop = generate_batch(x_test, [], [], y_pos_test,
                                                                                    x_mask_test, sense_mask_test,
                                                                                    train=False, test_ids=x_ids_test)
                        _logits = sess.run(logits_sen, feed_dict={x: bx, y_pos: by_pos, sense_mask: bs_mask,
                                                                  x_mask: bx_mask, embeddings: self.embeddings})

                        for i in range(self.batch_size):
                            for j in range(window_size):
                                if bx_mask[i, j]:
                                    if bs_mask[i, j]:
                                        if bx[i, j] not in self.possible_senses:
                                            predicted = self.MFS.get(self.id2word[bx[i, j]], "bn:00000000x")
                                        else:
                                            possible_logits = []
                                            possible = self.filter_senses(bx[i, j], by_pos[i, j])
                                            for s in possible:
                                                possible_logits.append(_logits[i, j, s])
                                            predicted = self.id2sense[possible[int(np.argmax(possible_logits))]]

                                        print("{}\t{}".format(bx_ids[i, j], predicted), file=f)

        return

    def filter_senses(self, word_id, pos):
        """
        Check which possible senses of a word are compatible with the
        given part of speech using the last letter of the sense ID string.
        :param word_id: index of a word in the dictionary
        :param pos: part of speech index
        :return: list of possible sense indices for given word and POS
        """
        sense_ids = self.possible_senses[word_id]
        result = []
        for sense in sense_ids:
            if self.id2pos[pos] == "NOUN":
                if self.id2sense[sense][-1:] == "n":
                    result.append(sense)
            if self.id2pos[pos] == "VERB":
                if self.id2sense[sense][-1:] == "v":
                    result.append(sense)
            if self.id2pos[pos] == "ADJ":
                if self.id2sense[sense][-1:] == "a":
                    result.append(sense)
            if self.id2pos[pos] == "ADV":
                if self.id2sense[sense][-1:] == "r":
                    result.append(sense)
        if len(result) == 0:
            result.append(self.sense2id.get(self.MFS.get(self.id2word[word_id], "bn:00000000x"), 25913))
            # or better: most frequent sense
        return result

