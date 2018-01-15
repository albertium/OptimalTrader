
import numpy as np


def continuous_converter(params):
    _min = params["min"]
    _const = 1 / (params["max"] - _min) * params["num_grids"]
    return lambda x: np.floor((np.array(x) - _min) * _const).astype(int)


def discrete_converter(params):
    _min = params["min"]
    return lambda x: (np.array(x) - _min).astype(int)


def incremental_constraint(params):
    _min, _max = params["min"], params["max"]
    return lambda x: (_min - x, _max - x)


def direct_constraint(params):
    _min, _max = params["min"], params["max"]
    return lambda x: (_min, _max)


node_types = {
    "continuous": {
        "required_params": ["min", "max", "num_grids"],
        "dimension_func": lambda params: params["num_grids"],
        "converter": continuous_converter
    },
    "discrete": {
        "required_params": ["min", "max"],
        "dimension_func": lambda params: params["max"] - params["min"] + 1,
        "converter": discrete_converter
    }
}

link_types = {
    "incremental": {
        "action_constraint": incremental_constraint,
        "mover": lambda action, state: state + action
    },
    "direct": {
        "action_constraint": direct_constraint,
        "mover": lambda action, state: action
    }
}


