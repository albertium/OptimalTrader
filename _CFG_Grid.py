
import numpy as np
import math


# ----- converters -----
def continuous_converter(params):
    _min = params["min"]
    _const = 1 / (params["max"] - _min) * params["cells"]
    return lambda x: math.floor((float(x) - _min) * _const)


def discrete_converter(params):
    _min = params["min"]
    return lambda x: int(x) - _min


# ----- generators -----
def continuous_generator(params):
    _min, _max = params["min"], params["max"]
    _const = _max - _min
    return lambda: _min + _const * np.random.rand()


def discrete_generator(params):
    _min, _max = params["min"], params["max"]
    return lambda: np.random.randint(_min, _max + 1)


# ----- constraints -----
def incremental_constraint(params):
    _min, _max = params["min"], params["max"]
    return lambda x: (_min - x, _max - x)


def direct_constraint(params):
    _min, _max = params["min"], params["max"]
    return lambda x: (_min, _max)


node_types = {
    "continuous": {
        "dimension_func": lambda params: params["cells"],
        "converter": continuous_converter,
        "generator": continuous_generator
    },
    "discrete": {
        "dimension_func": lambda params: params["max"] - params["min"] + 1,
        "converter": discrete_converter,
        "generator": discrete_generator
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


