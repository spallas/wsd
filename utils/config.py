from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Config(object):

    batch_size: int = 32
    accumulation_steps: int = 1
    num_epochs: int = 40
    window_size: int = 100
    mixed_precision: str = 'O0'
    multi_gpu: bool = False

    train_data: str = 'res/wsd-train/semcor+glosses_data.xml'
    train_tags: str = 'res/wsd-train/semcor+glosses_tags.txt'
    eval_data:  str = 'res/wsd-test/se07/se07.xml'
    eval_tags:  str = 'res/wsd-test/se07/se07.txt'
    test_data:  str = 'res/wsd-train/test_data.xml'
    test_tags:  str = 'res/wsd-train/test_tags.txt'
    sense_dict: str = 'res/dictionaries/senses.txt'
    pad_symbol: str = '<pad>'

    log_interval: int = 400
    cache_embeddings: bool = False
    cache_path: str = 'res/cache.npz'
    embed_model_path: str = 'res/roberta.large'


@dataclass_json
@dataclass
class ElmoConfig(Config):

    hidden_size: int = 1024
    num_layers:  int = 2

    elmo_weights: str = ''
    elmo_options: str = ''
    elmo_size: int = ''

    learning_rate: float = 0.001

    checkpoint_path: str = 'saved_weights/baseline_elmo_checkpoint.pt'
    report_path: str = 'logs/baseline_elmo_report.txt'

    @staticmethod
    def from_json_file(file_name, **kwargs):
        with open(file_name) as f:
            return ElmoConfig.from_json(f.read(), **kwargs)


@dataclass_json
@dataclass
class BertTransformerConfig(Config):

    checkpoint_path: str = 'saved_weights/bert_wsd_checkpoint.pt'
    report_path: str = 'logs/bert_wsd_report.txt'

    learning_rate: float = 0.00005
    d_model: int = 512
    num_layers: int = 2
    num_heads: int = 4
    pos_embed_dim: int = 32

    bert_model = 'bert-large-cased'

    @staticmethod
    def from_json_file(file_name, **kwargs):
        with open(file_name) as f:
            return BertTransformerConfig.from_json(f.read(), **kwargs)


@dataclass_json
@dataclass
class ElmoTransformerConfig(Config):

    checkpoint_path: str = 'saved_weights/elmo_tr_checkpoint.pt'
    report_path: str = 'logs/elmo_tr_report.txt'

    elmo_weights: str = ''
    elmo_options: str = ''
    elmo_size: int = 0

    learning_rate: float = 0.00005
    d_model: int = 512
    pos_embed_dim: int = 32
    num_heads: int = 4

    @staticmethod
    def from_json_file(file_name, **kwargs):
        with open(file_name) as f:
            return ElmoTransformerConfig.from_json(f.read(), **kwargs)


@dataclass_json
@dataclass
class RobertaTransformerConfig(Config):

    checkpoint_path: str = 'saved_weights/roberta_tr_checkpoint.pt'
    report_path: str = 'logs/roberta_tr_report.txt'

    model_path: str = 'res/roberta.large'
    learning_rate: float = 0.00005
    d_embeddings: int = 1024
    d_model: int = 512
    pos_embed_dim: int = 32
    num_heads: int = 4
    num_layers: int = 6

    @staticmethod
    def from_json_file(file_name, **kwargs):
        with open(file_name) as f:
            return RobertaTransformerConfig.from_json(f.read(), **kwargs)


@dataclass_json
@dataclass
class RDenseConfig(Config):

    checkpoint_path: str = 'saved_weights/r_dense_checkpoint.pt'
    report_path: str = 'logs/r_dense_report.txt'
    model_path: str = 'res/roberta.large'
    learning_rate: float = 0.0001
    d_embeddings: int = 1024
    hidden_dim: int = 512

    @staticmethod
    def from_json_file(file_name, **kwargs):
        with open(file_name) as f:
            return RDenseConfig.from_json(f.read(), **kwargs)


@dataclass_json
@dataclass
class WSDNetXConfig(RobertaTransformerConfig):
    output_vocab: str = 'res/dictionaries/syn_lemma_vocab.txt'
    sense_lemmas: str = 'res/dictionaries/sense_lemmas.txt'

    @staticmethod
    def from_json_file(file_name, **kwargs):
        with open(file_name) as f:
            return WSDNetXConfig.from_json(f.read(), **kwargs)


@dataclass_json
@dataclass
class WSDDenseConfig(Config):

    checkpoint_path: str = 'saved_weights/wsd_dense_checkpoint.pt'
    report_path: str = 'logs/wsd_dense_report.txt'
    model_path: str = 'res/roberta.large'
    learning_rate: float = 0.0001
    d_embeddings: int = 1024
    hidden_dim: int = 512

    output_vocab: str = 'res/dictionaries/syn_lemma_vocab.txt'
    sense_lemmas: str = 'res/dictionaries/sense_lemmas.txt'

    @staticmethod
    def from_json_file(file_name, **kwargs):
        with open(file_name) as f:
            return WSDDenseConfig.from_json(f.read(), **kwargs)


# Test
if __name__ == "__main__":
    c = WSDNetXConfig.from_json_file("../conf/wsdnetx_3280.json")
    print(c.to_json())
