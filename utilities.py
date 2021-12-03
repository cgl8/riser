import yaml
from attrdict import AttrDict

# TODO: Verify config
def get_config(filepath):
    with open(filepath) as config_file:
        return AttrDict(yaml.load(config_file, Loader=yaml.Loader))
