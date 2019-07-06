from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Config(object):

    batch_size: int = 32
    num_epochs: int = 40

    data_path: str = 'res/wsd-train/semcor_data.xml'
    tags_path: str = 'res/wsd-train/semcor_tags.txt'

    dev_data_path: str = ''
    dev_tags_path: str = ''

    test_data_path: str = ''
    test_tags_path: str = ''


@dataclass_json
@dataclass
class ElmoConfig(Config):

    hidden_size: int = 1024
    num_layers: int = 2

    elmo_weights: str = ''
    elmo_options: str = ''
    elmo_size: int = ''

    learning_rate: float = 0.001

    checkpoint_path: str = ''
    logs_path: str = ''


@dataclass_json
@dataclass
class WSDNetConfig(Config):

    learning_rate: float = 0.001

    checkpoint_path: str = ''
    logs_path: str = ''


# Test
if __name__ == "__main__":
    with open("conf/elmo_conf.json") as f:
        c = ElmoConfig.from_json(f.read())
    print(c.to_json())
