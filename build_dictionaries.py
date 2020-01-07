import argparse
from collections import OrderedDict
from nltk.corpus import wordnet as wn


def _get_dicts(syn_lemma_vocab='res/dictionaries/syn_lemma_vocab.txt',
               senses_vocab='res/dictionaries/senses.txt'):
    with open(senses_vocab) as f:
        sense2id = {line.strip().split()[0]: int(line.strip().split()[1]) for line in f}
    wn_lemmas = OrderedDict()
    with open(syn_lemma_vocab) as f:
        for i, line in enumerate(f):
            wn_lemmas[line.strip()] = i
    return sense2id, wn_lemmas


def build_1hop(dest_path='res/dictionaries/sense_lemmas_1hop.txt',
               syn_lemma_vocab='res/dictionaries/syn_lemma_vocab.txt',
               senses_vocab='res/dictionaries/senses.txt'):
    sense2id, wn_lemmas = _get_dicts(syn_lemma_vocab, senses_vocab)
    with open(dest_path, 'w') as f:
        for S in sorted(sense2id.keys()):
            synset = wn.synset(S)
            hyper_list = synset.hypernyms()
            hypo_list = synset.hyponyms()
            synset_lemmas = synset.lemma_names()
            for h_s in hyper_list + hypo_list:
                synset_lemmas += h_s.lemma_names()
            syn_lemma_ids = [wn_lemmas[l] for l in synset_lemmas if l in wn_lemmas]
            print(f"{sense2id[S]}\t{syn_lemma_ids}", file=f)

#
# def build_blc(dest_path='res/dictionaries/sense_lemmas_blc.txt',
#               syn_lemma_vocab='res/dictionaries/syn_lemma_vocab.txt',
#               senses_vocab='res/dictionaries/senses.txt'):
#     sense2id, wn_lemmas = _get_dicts(syn_lemma_vocab, senses_vocab)
#     smap = SynsetMap('blc', 'res/blc/???')
#     with open(dest_path, 'w') as f:
#         for S in sorted(sense2id.keys()):
#             actual_synset = smap.map_synset(S)
#             syn_lemma_ids = [wn_lemmas[l] for l in actual_synset.lemma_names() if l in wn_lemmas]
#             print(f"{sense2id[S]}\t{syn_lemma_ids}", file=f)
#
#
# def build_macro(dest_path='res/dictionaries/sense_lemmas_macro.txt',
#                 syn_lemma_vocab='res/dictionaries/syn_lemma_vocab.txt',
#                 senses_vocab='res/dictionaries/senses.txt'):
#     sense2id, wn_lemmas = _get_dicts(syn_lemma_vocab, senses_vocab)
#     smap = SynsetMap('macro', 'res/macrosenses/???')
#     with open(dest_path, 'w') as f:
#         for S in sorted(sense2id.keys()):
#             actual_synset = smap.map_synset(S)
#             syn_lemma_ids = [wn_lemmas[l] for l in actual_synset.lemma_names() if l in wn_lemmas]
#             print(f"{sense2id[S]}\t{syn_lemma_ids}", file=f)


def build_hyper(dest_path='res/dictionaries/sense_lemmas_hyper.txt',
                syn_lemma_vocab='res/dictionaries/syn_lemma_vocab.txt',
                senses_vocab='res/dictionaries/senses.txt'):
    sense2id, wn_lemmas = _get_dicts(syn_lemma_vocab, senses_vocab)
    with open(dest_path, 'w') as f:
        for S in sorted(sense2id.keys()):
            synset = wn.synset(S)
            hyper_list = synset.hypernyms()
            synset_lemmas = synset.lemma_names()
            for h_s in hyper_list:
                synset_lemmas += h_s.lemma_names()
            syn_lemma_ids = [wn_lemmas[l] for l in synset_lemmas if l in wn_lemmas]
            print(f"{sense2id[S]}\t{syn_lemma_ids}", file=f)


def build_synset(dest_path='res/dictionaries/sense_lemmas_synset_.txt',
                 syn_lemma_vocab='res/dictionaries/syn_lemma_vocab.txt',
                 senses_vocab='res/dictionaries/senses.txt'):
    sense2id, wn_lemmas = _get_dicts(syn_lemma_vocab, senses_vocab)
    with open(dest_path, 'w') as f:
        for S in sorted(sense2id.keys()):
            synset_lemmas = wn.synset(S).lemma_names()
            syn_lemma_ids = [wn_lemmas[l] for l in synset_lemmas]
            print(f"{sense2id[S]}\t{syn_lemma_ids}", file=f)


def main():
    build_hyper()


if __name__ == '__main__':
    main()
