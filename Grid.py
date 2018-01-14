
import numpy as np
from core import check


class ActionValueGrid:
    def __init__(self):
        self.state_types = ["_".join(s.split("_")[2:-1]) for s in dir(self)
                            if s.startswith("_add_") and s.endswith("_state")]

        self.states = {}
        self.grid_dimension = []
        self.state_converters = []

    def add_states(self, states):
        if isinstance(states, dict):
            states = [states]

        for state in states:
            check("name" in state, "Please specify name for state")
            check("state_type" in state, "Please specify state_type for state " + state["name"])
            check("params" in state, "Please provide params for state " + state["name"])
            check(state["state_type"] in self.state_types, "unknown state_type for state " + state["name"])

            add_state_func = getattr(self, "_add_{}_state".format(state["state_type"]))
            add_state_func(state)

    def _add_continuous_state(self, state):
        params = state["params"]
        check(len(params.keys() - {"min", "max", "num_grids"}) == 0,
              "At least 1 params is missing for state" + state["name"])

        self.grid_dimension += [params["num_grids"]]
        _min = params["min"]
        _const = 1 / (params["max"] - _min) * params["num_grids"]
        self.state_converters += lambda x: np.floor((x - _min) * _const).astype(int)
        self.states[state["name"]] = len(self.states)

    def _add_consecutive_discrete_state(self, state):
        params = state["params"]
        check(len(params.keys() - {"min", "max"}) == 0,
              "At least 1 params is missing for state" + state["name"])

        _min, _max = params["min"], params["max"]
        self.grid_dimension += [_max - _min + 1]
        self.state_converters += lambda x: x - _min
        self.states[state["name"]] = len(self.states)

    def get_grid_index(self):
        pass