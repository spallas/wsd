import pickle
import numpy as np
import xml.etree.ElementTree as Et
from collections import Counter

# ======== Please put all flags here ======== #

LSTM_SIZE = 256
WINDOW_SIZE = 64
OVERLAP_SIZE = 8

# Model type
USE_LEMMAS = True
USE_CNN = False
USE_DROPOUT = False
USE_ATTENTION = True
USE_SENSEMBED = True

BATCH_SIZE = 128
LEARNING_RATE = (0.001, 200, 0.99)  # initial, decay steps, decay rate
NUM_EPOCHS = 7
DROPOUT_KEEP_PROB = 0.3
CLIP_GRADS = True

GLOVE_LIMIT = 174558
VOCABULARY_SIZE = 50_000
EMBEDDING_SIZE = 300

TRAIN_DATA = "../semcor.data.xml"
GROUND_TRUTH = "../semcor.gold.key.bnids.txt"
VAL_DATA = "../ALL.data.xml"
VAL_TRUTH = "../ALL.gold.key.bnids.txt"
EMBEDDINGS_FILE = "../embeddings35M.pkl"
VOCABULARY_FILE = "../dataset35M.pkl"
GLOVE_FILE = "../glove-51k.txt"
SENSEMBED_FILTERED_FILE = "../sensembed_for_semcor.pkl"
TEST_DATA = "../test_data.txt"

# =========================================== #


def load_dataset():
    """
    This function does in the order: parse training and development dataset
    (namely the Semcor dataset); builds mappings between integers and POS, senses,
    words; loads the embeddings (GloVe or SensEmbed); fills uniform size numpy arrays
    with data from the dataset.
    :return: a dictionary with the following keys:
                train: a tuple of arrays with words, POS, sense, masks, and embeddings matrix
                dev_dict: the same as train but divided by corpus name, e.g. key = senseval2, val = (x, y, y_pos, ...)
                poss_dict: for each word seen during training gives a list of possible sense_ids for that word
                maps: the dictionaries built like mfs that returns the most frequent sense of a word.
    """
    docid2sense = {}
    ids = []
    senses = set()

    with open(GROUND_TRUTH) as f:
        for line in f:
            word_id, sense = line.strip().split(" ")
            ids.append(word_id)
            senses.add(sense)
            docid2sense[word_id] = sense

    dev_ids = []
    dev_senses = set()
    dev_docid2sense = {}

    with open(VAL_TRUTH) as f:
        for line in f:
            word_id, sense = line.strip().split(" ")
            dev_ids.append(word_id)
            dev_senses.add(sense)
            dev_docid2sense[word_id] = sense

    num_train_senses = len(senses)
    print("Total number of senses = {}".format(num_train_senses))

    sentences = []
    sentence_lengths = []
    sense_lists = []
    pos_lists = []
    sense_masks = []
    data_tree = Et.parse(TRAIN_DATA)
    corpus = data_tree.getroot()

    for text in corpus:
        for sentence in text:
            word_list = []
            sense_list = []
            pos_list = []
            sense_mask = []  # True when parsing instance, False otherwise
            for word in sentence:
                if USE_LEMMAS:
                    word_list.append(word.attrib["lemma"])
                else:
                    word_list.append(word.text.lower())
                pos_list.append(word.attrib["pos"])
                if word.tag == "instance":
                    sense_mask.append(True)
                    sense = docid2sense[word.attrib["id"]]
                    sense_list.append(sense)
                else:
                    sense_mask.append(False)
                    sense_list.append("bn:00000000x")
            sentences.append(word_list)
            sense_lists.append(sense_list)
            pos_lists.append(pos_list)
            sense_masks.append(sense_mask)
            sentence_lengths.append(len(word_list))
    max_sentence_len = max(sentence_lengths)
    print("Max sentence len: ", max_sentence_len)
    print("Number of sentences: ", len(sentences))
    print("Number of words: ", sum(sentence_lengths))

    # build sense vocabulary:
    id2sense = {}
    sense2id = {}
    for ind, sense in enumerate(senses):
        id2sense[ind] = sense
        sense2id[sense] = ind
    id2sense[num_train_senses] = "bn:00000000x"
    sense2id["bn:00000000x"] = num_train_senses

    # build POS vocabulary
    id2pos = {0: "NOUN", 1: "VERB", 2: "ADJ", 3: "ADV", 4: "PRON", 5: "ADP",
              6: "DET", 7: "CONJ", 8: "PRT", 9: "NUM", 10: "X", 11: "."}
    pos2id = {v: k for k, v in id2pos.items()}

    if USE_SENSEMBED:
        global EMBEDDING_SIZE
        EMBEDDING_SIZE = 400
        with open(SENSEMBED_FILTERED_FILE, "rb") as f:
            word2id, embeddings_matrix, mfs_dict = pickle.load(f)
    else:
        word2id, _, embeddings_matrix = load_embeddings()
        my_word2id = {"UNK": 0}
        my_embed_matrix = [[0 for _ in range(EMBEDDING_SIZE)]]
        for i in range(len(sentences)):
            for j in range(max_sentence_len):
                if j < sentence_lengths[i]:
                    if sentences[i][j] not in my_word2id:
                        if sentences[i][j] in word2id:
                            my_embed_matrix.append(embeddings_matrix[word2id[sentences[i][j]]])
                        else:
                            my_embed_matrix.append([0 for _ in range(EMBEDDING_SIZE)])
                        my_word2id[sentences[i][j]] = len(my_word2id)
        word2id, embeddings_matrix = my_word2id, my_embed_matrix

    global VOCABULARY_SIZE
    VOCABULARY_SIZE = len(word2id)
    print("My vocab size: ", VOCABULARY_SIZE)

    possible_senses = {}

    # prepare numpy arrays
    x_train = np.zeros((len(sentences), max_sentence_len), dtype=np.int32)
    y_train = np.zeros((len(sentences), max_sentence_len), dtype=np.int32)
    y_sen = np.zeros((len(sentences), max_sentence_len), dtype=np.int32)
    y_pos = np.zeros((len(sentences), max_sentence_len), dtype=np.int32)
    x_mask = np.zeros((len(sentences), max_sentence_len), dtype=np.bool)
    sense_mask_train = np.zeros((len(sentences), max_sentence_len), dtype=np.bool)
    sentence_lengths = np.array(sentence_lengths)

    for i in range(len(sentences)):
        for j in range(max_sentence_len):
            if j < sentence_lengths[i]:
                x_mask[i, j] = True
                sense_mask_train[i, j] = sense_masks[i][j]
                x_train[i, j] = word2id.get(sentences[i][j], 0)
                y_train[i, j] = x_train[i, j]  # auto-encoding
                y_sen[i, j] = sense2id.get(sense_lists[i][j])
                y_pos[i, j] = pos2id[pos_lists[i][j]]

                if x_train[i, j] in possible_senses:
                    possible_senses[x_train[i, j]].append(int(y_sen[i, j]))
                else:
                    possible_senses[x_train[i, j]] = [int(y_sen[i, j])]

    for k in possible_senses:
        possible_senses[k] = Counter(possible_senses[k])

    dev = {}
    dev_data_tree = Et.parse(VAL_DATA)
    dev_corpora = dev_data_tree.getroot()
    prev = "senseval2"
    dev_sentences = []
    dev_sentence_lengths = []
    dev_sense_lists = []
    dev_pos_lists = []
    dev_sense_masks = []

    for text in dev_corpora:
        corpus_name = text.attrib["id"][:-5]
        if corpus_name != prev:
            # prepare numpy arrays
            x_dev = np.zeros((len(dev_sentences), max_sentence_len), dtype=np.int32)
            y_dev = np.zeros((len(dev_sentences), max_sentence_len), dtype=np.int32)
            y_sen_dev = np.zeros((len(dev_sentences), max_sentence_len), dtype=np.int32)
            y_pos_dev = np.zeros((len(dev_sentences), max_sentence_len), dtype=np.int32)
            x_mask_dev = np.zeros((len(dev_sentences), max_sentence_len), dtype=np.bool)
            sense_mask_dev = np.zeros((len(dev_sentences), max_sentence_len), dtype=np.bool)

            for i in range(len(sentences)):
                for j in range(max_sentence_len):
                    if i < len(dev_sentences) and j < dev_sentence_lengths[i]:
                        x_mask_dev[i, j] = True
                        sense_mask_dev[i, j] = dev_sense_masks[i][j]
                        x_dev[i, j] = word2id.get(dev_sentences[i][j], 0)
                        y_dev[i, j] = word2id.get(dev_sentences[i][j], 0)
                        y_sen_dev[i, j] = sense2id.get(dev_sense_lists[i][j], num_train_senses)
                        # return index of unknown sense if the sense os not in the training dictionary
                        y_pos_dev[i, j] = pos2id[dev_pos_lists[i][j]]
            dev[prev] = (x_dev, y_dev, y_sen_dev, y_pos_dev, x_mask_dev, sense_mask_dev, dev_sentence_lengths)
            # get ready for next corpus
            dev_sentences = []
            dev_sentence_lengths = []
            dev_sense_lists = []
            dev_pos_lists = []
            dev_sense_masks = []
            prev = corpus_name

        for sentence in text:
            word_list = []
            sense_list = []
            pos_list = []
            sense_mask = []  # True when parsing instance, False otherwise
            for word in sentence:
                if USE_LEMMAS:
                    word_list.append(word.attrib["lemma"])
                else:
                    word_list.append(word.text.lower())
                pos_list.append(word.attrib["pos"])
                if word.tag == "instance":
                    sense_mask.append(True)
                    sense = dev_docid2sense[word.attrib["id"]]
                    sense_list.append(sense)
                else:
                    sense_mask.append(False)
                    sense_list.append("bn:00000000x")
            dev_sentences.append(word_list)
            dev_sense_lists.append(sense_list)
            dev_pos_lists.append(pos_list)
            dev_sense_masks.append(sense_mask)
            dev_sentence_lengths.append(len(word_list))

    # fill last corpus (namely semeval2015) before returning
    x_dev = np.zeros((len(dev_sentences), max_sentence_len), dtype=np.int32)
    y_dev = np.zeros((len(dev_sentences), max_sentence_len))
    y_sen_dev = np.zeros((len(dev_sentences), max_sentence_len))
    y_pos_dev = np.zeros((len(dev_sentences), max_sentence_len))
    x_mask_dev = np.zeros((len(dev_sentences), max_sentence_len), dtype=np.bool)
    sense_mask_dev = np.zeros((len(dev_sentences), max_sentence_len), dtype=np.bool)
    dev_sentence_lengths = np.array(dev_sentence_lengths)

    for i in range(len(sentences)):
        for j in range(max_sentence_len):
            if i < len(dev_sentences) and j < dev_sentence_lengths[i]:
                x_mask_dev[i, j] = True
                sense_mask_dev[i, j] = dev_sense_masks[i][j]
                x_dev[i, j] = word2id.get(dev_sentences[i][j], 0)
                y_dev[i, j] = word2id.get(dev_sentences[i][j], 0)
                y_sen_dev[i, j] = sense2id.get(dev_sense_lists[i][j], num_train_senses)
                # return index of unknown sense if the sense os not in the training dictionary
                y_pos_dev[i, j] = pos2id[dev_pos_lists[i][j]]
    print("Shuffling...")
    dev[prev] = shuffle_in_unison(x_dev, y_dev, y_sen_dev, y_pos_dev, x_mask_dev, sense_mask_dev, dev_sentence_lengths)
    x_train, y_train, y_sen, y_pos, x_mask, sense_mask_train, sentence_lengths = \
        shuffle_in_unison(x_train, y_train, y_sen, y_pos, x_mask, sense_mask_train, sentence_lengths)
    return {
        "train": (x_train, y_train, y_sen, y_pos, x_mask, sense_mask_train, embeddings_matrix, sentence_lengths),
        "dev_dict": dev,
        "poss_dict": possible_senses,
        "maps": (sense2id, id2sense, id2pos,
                 pos2id, word2id, mfs_dict) if USE_SENSEMBED else (sense2id, id2sense, id2pos, pos2id, word2id)
    }


def shuffle_in_unison(a, b, c, d, e, f, g):
    """
    :param a: array
    etc..
    :return: shuffled arrays
    """
    assert len(a) == len(b) == len(c) == len(d) == len(e) == len(f) == len(g)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    shuffled_c = np.empty(c.shape, dtype=c.dtype)
    shuffled_d = np.empty(d.shape, dtype=d.dtype)
    shuffled_e = np.empty(e.shape, dtype=e.dtype)
    shuffled_f = np.empty(f.shape, dtype=f.dtype)
    shuffled_g = np.empty(g.shape, dtype=g.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
        shuffled_c[new_index] = c[old_index]
        shuffled_d[new_index] = d[old_index]
        shuffled_e[new_index] = e[old_index]
        shuffled_f[new_index] = f[old_index]
        shuffled_g[new_index] = g[old_index]
    return shuffled_a, shuffled_b, shuffled_c, shuffled_d, shuffled_e, shuffled_f, shuffled_g


sent_i = 0  # index in the array of sentences
word_j = 0  # index of words in a sentence
reset = False


def generate_batch(x, y, y_sen, y_pos, x_mask, sense_mask, train=True, test_ids=None):
    """
    Returns a batch of dimension BATCH_SIZE, WINDOW_SIZE. Minimizes the padding used in training.
    If train is false then the overlapping across batches is ignored. If test ids is given it means
    the function is being used for testing so no senses are given and no auto-encoding is needed.
    :param x: array of sentences; each sentence is an array of word ids padded to the max sentence length
    :param y: the same array as x
    :param y_sen: matrix of sense indices where corresponding sense_mask is True.
    :param y_pos: parts of speech indices of each word
    :param x_mask: boolean mask with as many True values as the length of the corresponding sentence, False otherwise
    :param sense_mask: True if the word has been marked ambiguous
    :param train: flag set to False when evaluating or predicting.
    :param test_ids: ids of words in the test dataset
    :return: batched data + a reset flag set to True when the input data has been seen once
    """
    global sent_i
    global word_j
    global reset
    if reset:
        sent_i = word_j = 0
        reset = False
    batch_x = np.zeros((BATCH_SIZE, WINDOW_SIZE))
    batch_y = np.zeros((BATCH_SIZE, WINDOW_SIZE))
    batch_y_sen = np.zeros((BATCH_SIZE, WINDOW_SIZE))
    batch_y_pos = np.zeros((BATCH_SIZE, WINDOW_SIZE))
    if test_ids is not None:
        batch_test_ids = np.zeros((BATCH_SIZE, WINDOW_SIZE), dtype=test_ids.dtype)
    batch_x_mask = np.zeros((BATCH_SIZE, WINDOW_SIZE), dtype="bool")
    batch_sense_mask = np.zeros((BATCH_SIZE, WINDOW_SIZE), dtype="bool")
    for i in range(BATCH_SIZE):
        for j in range(WINDOW_SIZE):
            if sent_i < len(x) and word_j < len(x[0]):
                batch_x[i, j] = x[sent_i, word_j]
                if test_ids is None:  # not testing
                    batch_y[i, j] = y[sent_i, word_j]
                    batch_y_sen[i, j] = y_sen[sent_i, word_j]
                else:
                    batch_test_ids[i, j] = test_ids[sent_i, word_j]
                batch_y_pos[i, j] = y_pos[sent_i, word_j]
                batch_x_mask[i, j] = x_mask[sent_i, word_j]
                batch_sense_mask[i, j] = sense_mask[sent_i, word_j]
                word_j = word_j + 1
                if word_j == len(x[0]) - 1:
                    break
        if word_j == len(x[0]) - 1 or not x_mask[sent_i, word_j]:
            word_j = 0
            sent_i += 1
            if sent_i == len(x):
                reset = True
                break
            if sent_i % 1000 == 0:
                print("Sentence #:{}".format(sent_i))
        elif train:  # don't overlap in evaluation
            word_j -= OVERLAP_SIZE
            if word_j < 0:
                word_j = 0
    if test_ids is None:
        batch = batch_x, batch_y, batch_y_sen, batch_y_pos, batch_x_mask, batch_sense_mask, reset
    else:
        batch = batch_x, batch_test_ids, batch_y_pos, batch_x_mask, batch_sense_mask, reset
    return batch


def load_embeddings(from_glove=True):
    """
    Load from specified file and preprocess.
    :param from_glove: whether to use GloVe files or custom ones
    :return: integer to strings maps + the embeddings matrix
    """
    print("Loading embeddings...")
    word2id = {}
    id2word = {}

    if from_glove:
        glove = np.loadtxt(GLOVE_FILE, dtype='str', comments=None)
        words = glove[:GLOVE_LIMIT, 0]
        embeddings_matrix = glove[:GLOVE_LIMIT, 1:].astype('float')
        for w in words:
            id2word[len(id2word)] = w
            word2id[w] = len(id2word) - 1
    else:
        # load words vocabularies
        with open(VOCABULARY_FILE, "rb") as f:
            _, word2id, id2word, _ = pickle.load(f)
            del _
        with open(EMBEDDINGS_FILE, "rb") as f:
            embeddings_matrix = pickle.load(f)

    return word2id, id2word, embeddings_matrix


def load_test_set(pos2id, word2id):

    x_test = []
    x_mask_test = []
    x_ids_test = []
    y_pos_test = []
    y_sen_mask = []

    lengths = []
    with open(TEST_DATA) as f:
        for line in f:
            tokens = line.split()
            sentence = []
            mask = []
            ids = []
            pos_list = []
            sense_mask = []
            for t in tokens:
                word_info = t.split("|")
                if len(word_info) == 4:
                    word, lemma, pos, word_id = word_info
                    ids.append(word_id)
                    sense_mask.append(True)
                else:
                    word, lemma, pos = word_info
                    ids.append("-")
                    sense_mask.append(False)
                if USE_LEMMAS:
                    sentence.append(lemma.lower())
                else:
                    sentence.append(word.lower())
                pos_list.append(pos)
                mask.append(True)
            x_test.append(sentence)
            x_mask_test.append(mask)
            x_ids_test.append(ids)
            y_pos_test.append(pos_list)
            y_sen_mask.append(sense_mask)
            lengths.append(len(sentence))
    max_len = max(lengths)
    x_test = np.array([[word2id.get(i, 0) for i in xt] + [word2id["UNK"]]*(max_len-len(xt)) for xt in x_test])
    x_mask_test = np.array([xm+[False]*(max_len-len(xm)) for xm in x_mask_test])
    x_ids_test = np.array([xid+["-"]*(max_len-len(xid)) for xid in x_ids_test])
    y_pos_test = np.array([[pos2id[j] for j in yp] + [pos2id["X"]]*(max_len-len(yp)) for yp in y_pos_test])
    y_sen_mask = np.array([ys+[False]*(max_len-len(ys)) for ys in y_sen_mask])
    return x_test, x_ids_test, x_mask_test, y_pos_test, y_sen_mask


def from_sensembed(sentences):
    """
    From a list of sentences builds an ad-hoc dictionary and embedding retrieving the data
    from SensEmbed. Saves everything to a Pickle file.
    :param sentences: List of words to be used
    :return: Nothing; meant to be done offline.
    """
    def safe_to_float(float_str):
        try:
            n = float(float_str)
        except ValueError:
            print("hey! ", float_str)
            n = 0
        return n

    global EMBEDDING_SIZE
    EMBEDDING_SIZE = 400
    w2i_sen = {"UNK": 0}
    monosemous = {}
    sensembed_matrix = [[0 for _ in range(EMBEDDING_SIZE)]]
    semcor_vocab = set()

    for i in range(len(sentences)):
        for word in sentences[i]:
            semcor_vocab.add(word)
    print(len(semcor_vocab))

    mfs_dict = {"UNK": "bn:00000000x"}

    with open("../babelfy_vectors") as f:
        for line in f:
            w, vec_str = line.strip().split(maxsplit=1)
            if "_" not in w:
                if w in semcor_vocab:
                    if w not in monosemous:
                        monosemous[w] = vec_str
                continue
            w, mfs = w.rsplit("_", 1)
            if w in semcor_vocab:
                if w not in w2i_sen:
                    w2i_sen[w] = len(w2i_sen)
                    sensembed_matrix.append(list(map(safe_to_float, vec_str.split())))
                    mfs_dict[w] = mfs

    for w in semcor_vocab:
        if w not in w2i_sen:
            w2i_sen[w] = len(w2i_sen)
            if w in monosemous:
                sensembed_matrix.append(list(map(safe_to_float, monosemous[w].split())))
            else:
                sensembed_matrix.append([0 for _ in range(EMBEDDING_SIZE)])

    with open(SENSEMBED_FILTERED_FILE, "wb") as f:
        pickle.dump([w2i_sen, sensembed_matrix, mfs_dict], f)
    return


def get_params_dict():
    """
    To be used in the Model constructor to retrieve all the hyper-parameters
    :return: dictionary of hyper-parameters
    """
    return {
        "input_dropout": DROPOUT_KEEP_PROB,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "lstm_size": LSTM_SIZE,
        "vocab_size": VOCABULARY_SIZE,
        "embed_size": EMBEDDING_SIZE,
        "window_size": WINDOW_SIZE,
        "clip_grads": CLIP_GRADS,
        "model": (USE_CNN, USE_DROPOUT, USE_ATTENTION)
    }
