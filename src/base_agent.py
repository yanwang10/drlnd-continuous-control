import copy
import json

class BaseAgent:
    """
    A base class of agents with basic functionalities on configuration.
    """
    def __init__(self, config, default):
        self.config = copy.deepcopy(config)
        self.default = copy.deepcopy(default)

    def get(self, k):
        if k in self.config:
            return self.config[k]
        elif k in self.default:
            return self.default[k]
        else:
            return None

    def print_config(self):
        config = self.default
        for k in self.config:
            config[k] = self.config[k]
        print('Configs: ', json.dumps(config, indent=4))
