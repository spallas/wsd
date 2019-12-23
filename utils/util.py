import random
from collections import defaultdict
from typing import List

import telegram
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer
random.seed(42)

# MAPS for Part Of Speech #############

with open('data/bot_token.txt') as f:
    token = f.read().strip()
bot = telegram.Bot(token=token)
chat_id = 105475495

tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

univ_tag_map = defaultdict(lambda: "O")
univ_tag_map["VERB"] = "V"
univ_tag_map["NOUN"] = "N"
univ_tag_map["ADJ"] = "A"
univ_tag_map["ADV"] = "R"

pos2id = defaultdict(lambda: 0)
pos2id["VERB"] = 1
pos2id["NOUN"] = 2
pos2id["ADJ"] = 3
pos2id["ADV"] = 4

id2wnpos = defaultdict(lambda: 'n')
id2wnpos[1] = 'v'
id2wnpos[2] = 'n'
id2wnpos[3] = 'a'
id2wnpos[4] = 'r'

NOT_AMB_SYMBOL = 0
UNK_SENSE = -2
PAD_SYMBOL = '<pad>'

#######################################


def randomized(iterable):
    cache = [i for i in iterable]
    rand_indices = list(range(len(cache)))
    random.shuffle(rand_indices)
    for i in rand_indices:
        yield cache[i]


class Randomized:

    def __init__(self, iterable):
        self.cache = [e for e in iterable]
        self.rand_indices = list(range(len(self.cache)))
        self.i = 0

    def __iter__(self):
        random.shuffle(self.rand_indices)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.cache):
            raise StopIteration
        c = self.cache[self.rand_indices[self.i]]
        self.i += 1
        return c

def telegram_on_failure(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except Exception as e:
        bot.send_message(chat_id=chat_id,
                         text=f'ERROR!\n{e}')
        raise e


def telegram_result_value(function, *args, **kwargs):
    return_val = function(*args, **kwargs)
    try:
        bot.send_message(chat_id=chat_id, text=f'{return_val}')
    except Exception:
        return return_val
    return return_val


def telegram_send(message: str):
    try:
        bot.send_message(chat_id=chat_id, text=message)
    except Exception:
        pass


def example_to_input(lemma_list: List[str],
                     tags_list: List[int],
                     tok: BertTokenizer):
    subword_list, tags_map = tok.convert_tokens_to_ids(tok.tokenize('[CLS]')), []
    for w in lemma_list:
        tags_map.append(len(subword_list))
        subword_list += tok.convert_tokens_to_ids(tok.tokenize(w))
    subword_list += tok.convert_tokens_to_ids(tok.tokenize('[SEP]'))
    mapped_tags = [0] * len(subword_list)
    # mapped_pos = [0] * len(subword_list)
    # mapped_lemmas = ["[UNK]"] * len(subword_list)
    # mapped_altern = [[]] * len(subword_list)
    for i, j in enumerate(tag_map):
        mapped_tags[j] = tags_list[i]
        # mapped_pos[j] = example['pos'][i]
        # mapped_lemmas[j] = example['lemmas'][i]
        # mapped_altern[j] = example['alternatives'][i]
    return subword_list, mapped_tags


def is_ascii(s):
    return all(ord(c) < 128 for c in s)
