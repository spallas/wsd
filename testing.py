
from fairseq.models.roberta import alignment_utils

from data_preprocessing import FlatLoader, FlatSemCorDataset, load_sense2id


def test1():
    pass


def test2():
    from fairseq.models.roberta import RobertaModel
    roberta = RobertaModel.from_pretrained('res/roberta.large', checkpoint_file='model.pt')
    roberta.eval()

    dataset = FlatSemCorDataset('res/wsd-test/se07/se07.xml', 'res/wsd-test/se07/se07.txt')
    loader = FlatLoader(dataset, 32, 100, 'PAD')
    sense2id = load_sense2id()
    pred, true, z = [], [], []
    for step, (b_x, b_p, b_y, b_z) in enumerate(loader):
        for seq in b_x:
            sent = ' '.join(seq)
            encoded = roberta.encode(sent)
            alignment = alignment_utils.align_bpe_to_words(roberta, encoded, seq)
            features = roberta.extract_features(encoded, return_all_hiddens=False)
            features = features.squeeze(0)
            aligned = alignment_utils.align_features_to_words(roberta, features, alignment)[1:-1]

            print(aligned.shape)
            print(len(seq))
    print('\nDone.')


if __name__ == '__main__':
    test2()
