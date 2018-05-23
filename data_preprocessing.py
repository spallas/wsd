import pickle
import numpy as np
import itertools
import xml.etree.ElementTree as Et


# ======== Please put all flags here ======== #
LSTM_SIZE = 128

VOCABULARY_SIZE = 50_000
EMBEDDING_SIZE = 200

TRAIN_DATA = "../semcor.data.xml"
GROUND_TRUTH = "../semcor.gold.key.bnids.txt"
EMBEDDINGS_FILE = "../embeddings35M.pkl"
VOCABULARY_FILE = "../dataset35M.pkl"

BATCH_SIZE = 64
LEARNING_RATE = (0.001, 5000, 0.96)  # initial, decay steps, decay rate

# =========================================== #

# self.x_train, self.y_train, self.y_pos, self.x_mask, self.sense_mask


def load_dataset():
    """
    Load and preprocess dataset with caching (pickle objects dump)
    :return: train and development X and Y
    """
    docid2sense = {}
    ids = []
    senses = set()

    # load words vocabularies
    with open(VOCABULARY_FILE, "rb") as f:
        _, word2id, id2word, _ = pickle.load(f)
        del _
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings_list = pickle.load(f)

    with open("../semcor.gold.key.bnids.txt") as f:
        for line in f:
            word_id, sense = line.strip().split(" ")
            ids.append(word_id)
            senses.add(sense)
            docid2sense[word_id] = sense

    num_train_senses = len(senses)
    print("data_preprocessing => total number of senses = {}".format(num_train_senses))

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
    print(max_sentence_len)

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

    # prepare numpy arrays
    x_train = np.zeros((len(sentences), max_sentence_len), dtype=np.int32)
    y_train = np.zeros((len(sentences), max_sentence_len))
    y_pos = np.zeros((len(sentences), max_sentence_len))
    x_mask = np.zeros((len(sentences), max_sentence_len), dtype=np.bool)
    sense_mask = np.zeros((len(sentences), max_sentence_len), dtype=np.bool)

    for i in range(len(sentences)):
        for j in range(max_sentence_len):
            if j < sentence_lengths[i]:
                x_mask[i, j] = True
                sense_mask[i, j] = sense_lists[i][j]
                x_train[i, j] = word2id.get(sentences[i][j].lower(), 0)
                y_train[i, j] = sense2id.get(sense_lists[i][j])
                y_pos[i, j] = pos2id[pos_lists[i][j]]

    return x_train, y_train, y_pos, x_mask, sense_mask, embeddings_list, sentence_lengths


def get_embeddings():
    return load_embeddings(EMBEDDINGS_FILE)


def load_embeddings(embeddings_file):
    """
    Load from specified file and preprocess.
    :return: np array with the embeddings matrix
    """
    with open(embeddings_file, "rb") as f:
        embeddings_list = pickle.load(f)

    return np.asarray(embeddings_list)


def get_params_dict():
    return {
        "input_dropout": 0.4,
        "learning_rate": (0.01, 5000, 0.999),
        "batch_size": 32,
        "lstm_size": LSTM_SIZE,
        "vocab_size": VOCABULARY_SIZE,
        "embed_size": EMBEDDING_SIZE,
    }
