
from nltk.corpus import wordnet as wn
from typing import Set
import numpy as np
from pytorch_transformers import BertForMaskedLM
from sklearn.metrics import f1_score
from scipy.special import softmax

from bert_only import *

all_syn_lemmas = {}
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=False)
language_model = BertForMaskedLM.from_pretrained(BERT_MODEL)
language_model.eval()

def select_senses(b_scores, b_str, b_pos, sense2id):

    def to_ids(synsets):
        return set([sense2id.get(x.name(), 0) for x in synsets]) - {0}

    def set2padded(s: Set[int]):
        arr = np.array(list(s))
        return np.pad(arr, (0, b_scores.shape[-1] - len(s)), 'edge')

    b_impossible_senses = []
    # we will set to 0 senses not in WordNet for given lemma.
    for i, sent in enumerate(b_str):
        impossible_senses = []
        for j, lemma in enumerate(sent):
            sense_ids = to_ids(wn.synsets(lemma, pos=util.id2wnpos[b_pos[i][j]]))
            # if lemma in self.train_sense_map:
            #     sense_ids &= set(self.train_sense_map[lemma])
            padded = set2padded(set(range(b_scores.shape[-1])) - sense_ids)
            impossible_senses.append(padded)
        b_impossible_senses.append(impossible_senses)
    b_scores = b_scores.cpu().numpy()
    b_impossible_senses = np.array(b_impossible_senses)
    np.put_along_axis(b_scores, b_impossible_senses, np.min(b_scores), axis=-1)
    return np.argmax(b_scores, -1).tolist()


def lm_select_senses(b_scores, b_str, b_pos, sense2id, train_sense_map, b_labels):

    def to_ids(synsets):
        return set([sense2id.get(x.name(), 0) for x in synsets]) - {0}

    def set2padded(s: Set[int]):
        arr = np.array(list(s))
        return np.pad(arr, (0, b_scores.shape[-1] - len(s)), 'edge')

    def get_lemmas(synset):
        lemmas = synset.lemma_names()
        for s in synset.hyponyms():
            lemmas += s.lemma_names()
        for s in synset.hypernyms():
            lemmas += s.lemma_names()
        return lemmas

    b_impossible_senses = []
    # we will set to 0 senses not in WordNet for given lemma.
    for i, sent in enumerate(b_str):
        impossible_senses = []
        for j, lemma in enumerate(sent):
            sense_ids = to_ids(wn.synsets(lemma, pos=util.id2wnpos[b_pos[i][j]]))
            padded = set2padded(set(range(b_scores.shape[-1])) - set(sense_ids))
            impossible_senses.append(padded)
        b_impossible_senses.append(impossible_senses)
    b_scores = b_scores.cpu().numpy()
    b_impossible_senses = np.array(b_impossible_senses)
    np.put_along_axis(b_scores, b_impossible_senses, np.min(b_scores), axis=-1)
    b_scores = softmax(b_scores, -1)

    # Update b_scores with geometric mean with language model score.
    for i, sent in enumerate(b_str):
        for k, w in enumerate(sent):
            if b_labels[i][k] != NOT_AMB_SYMBOL:  # i.e. sense tagged word
                # text = ['[CLS]'] + sent + ['[SEP]']
                # text[k+1] = '[MASK]'
                text = [] + sent
                text[k] = '[MASK]'
                tokenized_text = []
                for ww in text:
                    tokenized_text += bert_tokenizer.tokenize(ww)
                masked_index = tokenized_text.index('[MASK]')
                indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
                tokens_tensor = torch.tensor([indexed_tokens])
                outputs = language_model(tokens_tensor)
                predictions = outputs[0]
                probabilities = torch.nn.Softmax(dim=0)(predictions[0, masked_index])

                lm_ids, lm_scores = [], []
                net_score = {}
                for S in wn.synsets(w, pos=util.id2wnpos[b_pos[i][k]]):
                    if S.name() not in sense2id:
                        continue
                    if len(wn.synsets(w, pos=util.id2wnpos[b_pos[i][k]])) == 1:
                        continue
                    s_id = sense2id[S.name()]
                    if S not in all_syn_lemmas:
                        all_syn_lemmas[S] = get_lemmas(S)
                    syn_tok_ids = []
                    for lemma in all_syn_lemmas[S]:
                        tokenized = bert_tokenizer.tokenize(lemma)
                        tok_ids = bert_tokenizer.convert_tokens_to_ids(tokenized)
                        syn_tok_ids += tok_ids
                    top_k = torch.topk(probabilities[syn_tok_ids,], k=10)[0].tolist() \
                        if len(syn_tok_ids) > 10 else probabilities[syn_tok_ids,].tolist()
                    s_score = sum(top_k)
                    lm_ids.append(s_id)
                    lm_scores.append(s_score)
                    net_score[s_id] = b_scores[i, k, s_id]
                if not lm_scores:
                    continue
                lm_score = {k: v for k, v in zip(lm_ids, softmax(lm_scores, 0))}
                for s_id in lm_score:
                    if w in train_sense_map:
                        b_scores[i, k, s_id] = (net_score[s_id] * lm_score[s_id]) ** 0.5
                    else:
                        b_scores[i, k, s_id] = lm_score[s_id]

    return np.argmax(b_scores, -1).tolist()


def main():
    train_dataset = FlatSemCorDataset('res/wsd-train/semcor_data.xml', 'res/wsd-train/semcor_tags.txt')
    dataset = FlatSemCorDataset('res/wsd-test/se07/se07.xml', 'res/wsd-test/se07/se07.txt')
    loader = BertSimpleLoader(dataset, 33, 100)
    sense2id = load_sense2id()
    pred, true, z = [], [], []
    for step, (b_t, b_x, b_p, b_l, b_y, b_s, b_z) in enumerate(loader):
        scores = torch.rand([33, 102, dataset.num_tags])
        pred += [item for seq in lm_select_senses(scores, b_x, b_p, sense2id, train_dataset.train_sense_map, b_y) for item in seq]
        true += [item for seq in b_y.tolist() for item in seq]
        z += [item for seq in b_z for item in seq]

    true_eval, pred_eval = [], []
    for i in range(len(true)):
        if true[i] == NOT_AMB_SYMBOL:
            continue
        else:
            if pred[i] in z[i]:
                true_eval.append(pred[i])
            else:
                true_eval.append(true[i])
            pred_eval.append(pred[i])
    f1 = f1_score(true_eval, pred_eval, average='micro')
    print(f"F1 = {f1}")


if __name__ == '__main__':
    main()
