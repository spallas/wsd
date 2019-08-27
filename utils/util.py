
from collections import defaultdict
from typing import List

from nltk.corpus import wordnet as wn
from pytorch_pretrained_bert import BertTokenizer

# MAPS for Part Of Speech #############

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

NOT_AMB_SYMBOL = -1
UNK_SENSE = -2
PAD_SYMBOL = 'PAD'  # '<pad>'

#######################################


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