
from collections import defaultdict
from nltk.corpus import wordnet as wn

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

#######################################
