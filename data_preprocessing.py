import pickle
import numpy as np
import xml.etree.ElementTree as Et

from collections import Counter

# ======== Please put all flags here ======== #
LSTM_SIZE = 256
WINDOW_SIZE = 64
OVERLAP_SIZE = 0

VOCABULARY_SIZE = 50_000 # 174558  # GloVe
EMBEDDING_SIZE = 300  # GloVe

MY_VOCABULARY_SIZE = 50_000
MY_EMBED_SIZE = 200

TRAIN_DATA = "../semcor.data.xml"
GROUND_TRUTH = "../semcor.gold.key.bnids.txt"
EMBEDDINGS_FILE = "../embeddings35M.pkl"
VOCABULARY_FILE = "../dataset35M.pkl"
GLOVE_FILE = "../glove_word_embeds.txt"

BATCH_SIZE = 64
LEARNING_RATE = (0.001, 1000, 0.96)  # initial, decay steps, decay rate

EVAL_CORPORA = ["senseval2", "senseval3", "semeval2007", "semeval2013", "semeval2015"]

# =========================================== #


def load_dataset():
    """
    Load and preprocess dataset with caching (pickle objects dump)
    :return: train and development X and Y
    """
    docid2sense = {}
    ids = []
    senses = set()

    word2id, id2word, embeddings_matrix = load_embeddings()

    with open("../semcor.gold.key.bnids.txt") as f:
        for line in f:
            word_id, sense = line.strip().split(" ")
            ids.append(word_id)
            senses.add(sense)
            docid2sense[word_id] = sense

    dev_ids = []
    dev_senses = set()
    dev_docid2sense = {}

    with open("../ALL.gold.key.bnids.txt") as f:
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
    data_tree = Et.parse("../semcor.data.xml")
    corpus = data_tree.getroot()

    for text in corpus:
        for sentence in text:
            word_list = []
            sense_list = []
            pos_list = []
            sense_mask = []  # True when parsing instance, False otherwise
            for word in sentence:
                word_list.append(word.text)
                pos_list.append(word.attrib["pos"])
                if word.tag == "instance":
                    sense_mask.append(True)
                    sense = docid2sense[word.attrib["id"]]
                    sense_list.append(sense)
                else:
                    sense_mask.append(False)
                    # sense_list.append(word.text)
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
    id2pos = {0: "NOUN", 1: "VERB", 2: "ADJ", 3: "ADP", 4: "PRON", 5: "ADV",
              6: "DET", 7: "CONJ", 8: "PRT", 9: "NUM", 10: "X", 11: "."}
    pos2id = {v: k for k, v in id2pos.items()}

    possible_senses = {}

    # prepare numpy arrays
    x_train = np.zeros((len(sentences), max_sentence_len), dtype=np.int32)
    y_train = np.zeros((len(sentences), max_sentence_len), dtype=np.int32)
    y_sen = np.zeros((len(sentences), max_sentence_len), dtype=np.int32)
    y_pos = np.zeros((len(sentences), max_sentence_len), dtype=np.int32)
    x_mask = np.zeros((len(sentences), max_sentence_len), dtype=np.bool)
    sense_mask_train = np.zeros((len(sentences), max_sentence_len), dtype=np.bool)

    for i in range(len(sentences)):
        for j in range(max_sentence_len):
            if j < sentence_lengths[i]:
                x_mask[i, j] = True
                sense_mask_train[i, j] = sense_masks[i][j]
                x_train[i, j] = word2id.get(sentences[i][j].lower(), 0)
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
    dev_data_tree = Et.parse("../ALL.data.xml")
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
                        x_dev[i, j] = word2id.get(dev_sentences[i][j].lower(), 0)
                        y_dev[i, j] = word2id.get(dev_sentences[i][j].lower(), 0)
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
                word_list.append(word.text)
                pos_list.append(word.attrib["pos"])
                if word.tag == "instance":
                    sense_mask.append(True)
                    sense = dev_docid2sense[word.attrib["id"]]
                    sense_list.append(sense)
                else:
                    sense_mask.append(False)
                    # sense_list.append(word.text)
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

    for i in range(len(sentences)):
        for j in range(max_sentence_len):
            if i < len(dev_sentences) and j < dev_sentence_lengths[i]:
                x_mask_dev[i, j] = True
                sense_mask_dev[i, j] = dev_sense_masks[i][j]
                x_dev[i, j] = word2id.get(dev_sentences[i][j].lower(), 0)
                y_dev[i, j] = word2id.get(dev_sentences[i][j].lower(), 0)
                y_sen_dev[i, j] = sense2id.get(dev_sense_lists[i][j], num_train_senses)
                # return index of unknown sense if the sense os not in the training dictionary
                y_pos_dev[i, j] = pos2id[dev_pos_lists[i][j]]
    dev[prev] = (x_dev, y_dev, y_sen_dev, y_pos_dev, x_mask_dev, sense_mask_dev, dev_sentence_lengths)

    return {
        "train": (x_train, y_train, y_sen, y_pos, x_mask, sense_mask_train, embeddings_matrix, sentence_lengths),
        "dev_dict": dev,
        "poss_dict": possible_senses
    }


sent_i = 0  # index in the array of sentences
word_j = 0  # index of words in a sentence
reset = False


def generate_batch(x, y, y_sen, y_pos, x_mask, sense_mask, train=True):
    global sent_i
    global word_j
    global reset
    if reset:
        sent_i = word_j = 0
    batch_x = np.zeros((BATCH_SIZE, WINDOW_SIZE))
    batch_y = np.zeros((BATCH_SIZE, WINDOW_SIZE))
    batch_y_sen = np.zeros((BATCH_SIZE, WINDOW_SIZE))
    batch_y_pos = np.zeros((BATCH_SIZE, WINDOW_SIZE))
    batch_x_mask = np.zeros((BATCH_SIZE, WINDOW_SIZE), dtype="bool")
    batch_sense_mask = np.zeros((BATCH_SIZE, WINDOW_SIZE), dtype="bool")
    for i in range(BATCH_SIZE):
        for j in range(WINDOW_SIZE):
            if sent_i < len(x) and word_j < len(x[0]):
                batch_x[i, j] = x[sent_i, word_j]
                batch_y[i, j] = y[sent_i, word_j]
                batch_y_sen[i, j] = y_sen[sent_i, word_j]
                batch_y_pos[i, j] = y_pos[sent_i, word_j]
                batch_x_mask[i, j] = x_mask[sent_i, word_j]
                batch_sense_mask[i, j] = sense_mask[sent_i, word_j]
                word_j = (word_j + 1) % len(x[0])
                if word_j == len(x[0]) - 1:
                    break
        if train:  # don't overlap in evaluation
            word_j -= OVERLAP_SIZE
        if not x_mask[sent_i, word_j] or word_j == len(x[0]) - 1:
            sent_i += 1
            if sent_i == len(x) - 1:
                reset = True
                break

    return batch_x, batch_y, batch_y_sen, batch_y_pos, batch_x_mask, batch_sense_mask, reset


def load_embeddings(from_glove=True):
    """
    Load from specified file and preprocess.
    :return: np array with the embeddings matrix
    """
    print("Loading embeddings...")
    word2id = {}
    id2word = {}

    if from_glove:
        glove = np.loadtxt(GLOVE_FILE, dtype='str', comments=None)
        words = glove[:VOCABULARY_SIZE, 0]
        embeddings_matrix = glove[:VOCABULARY_SIZE, 1:].astype('float')
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


def get_params_dict():
    return {
        "input_dropout": 0.4,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "lstm_size": LSTM_SIZE,
        "vocab_size": VOCABULARY_SIZE,
        "embed_size": EMBEDDING_SIZE,
        "window_size": WINDOW_SIZE
    }
