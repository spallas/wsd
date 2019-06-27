# Neural Word Sense Disambiguation integrating Lexical Knowledge

## Intro

In this work we present a Word Sense Disambiguation (WSD) engine
that integrates a Transformer-based neural architecture with 
knowledge present in WordNet, the resource from which the sense 
inventory is taken from.

## Model

The architecture is composed of ELMo embeddings plus a TransformerXL (x3)
on top with a final dense layer for tagging each word with the
right lemma, pos, and sense identifier.

To incorporate lexical knowledge at evaluation time where we score 
each possible sense of a word with different scores:
- the semantic similarity of the context with the gloss of the sense and it's 
  direct hypernyms and hyponyms.
- the accumulated probabilities of BERT language model for the lemma names
  of the synset and of its direct hypernyms and hyponyms.

## Training data

As a training dataset we use both SemCor and the annotated glosses.

## Further Details

Please refer to the [wiki page](https://github.com/spallas/wsd/wiki) in this 
repository for further details about the implementation.
