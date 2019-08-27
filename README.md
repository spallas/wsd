# Neural Word Sense Disambiguation integrating Lexical Knowledge

## Intro

In this work we present a Word Sense Disambiguation (WSD) engine
that integrates a Transformer-based neural architecture with 
knowledge present in WordNet, the resource from which the sense 
inventory is taken from.

## Model

The architecture is composed of contextualized embeddings plus 
a Transformer on top with a final dense layer.

To incorporate lexical knowledge at evaluation time where we score 
each possible sense of a word with a Language-Model-derived score.

To calculate this score we take for each possible synset the lemmas
of the synset and we extend this list with the lemmas of the direct
hypernyms and hyponyms (also 'related' ones?). Then for each lemma 
in the synset list we calculate the probability from the language
model for that lemma to appear as the next word in the text.
To avoid having synset with longer lists score higher we consider 
the following ways of aggregating the scores of each lemma in the 
list:
- max
- mean
- sum top 5
- mean top 5

## Training data

As a training dataset we use both SemCor and the annotated glosses.

## Further Details

Please refer to the [wiki page](https://github.com/spallas/wsd/wiki) in this 
repository for further details about the implementation.


## Notes: RoBERTa installation

```
# Download roberta.large model
cd res/
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
tar -xzvf roberta.large.tar.gz
```

```
# Load the model in fairseq
from fairseq.models.roberta import RobertaModel
roberta = RobertaModel.from_pretrained('res/roberta.large', checkpoint_file='model.pt')
roberta.eval()  # disable dropout (or leave in train mode to finetune)
```