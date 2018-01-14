
import numpy as np
from core import check


class ActionValueGrid:
    def __init__(self):
        self.types = ["_".join(s.split("_")[2:]) for s in dir(self) if s.startswith("_add_")]
        self.link_types = ["_".join(s.split("_")[2:]) for s in dir(self) if s.startswith("_link_")]

        self.states = {}
        self.grid_dimensions = []
        self.state_converters = []
        self.linked_states = []
        self.grid = None

        self.action_converter = None
        self.action_constraints = []

        # training parameters
        self.alpha = 0.001
        self.lam = 0.999
        self.eta = 0.1

    def add_action(self, actions):
        check("type" in actions, "Please specify type for action")
        check("params" in actions, "Please specify params for action")
        actions["name"] = "action"
        self.add_states(actions)

    def add_states(self, states):
        if isinstance(states, dict):
            states = [states]

        for state in states:
            check("name" in state, "Please specify name for state")
            check("type" in state, "Please specify type for state " + state["name"])
            check("params" in state, "Please provide params for state " + state["name"])
            check(state["type"] in self.types, "unknown type for state " + state["name"])

            add_state_func = getattr(self, "_add_{}".format(state["type"]))
            add_state_func(state)

    def link_action_to_state(self, states):
        if isinstance(states, dict):
            states = [states]

        for state in states:
            check("name" in state, "Please specify the state to link")
            check("type" in state, "Please specify type for the link to state " + state["name"])
            check(state["type"] in self.link_types, "unknown link type for state " + state["name"])

            link_func = getattr(self, "_link_{}".format(state["type"]))
            link_func(state)

    def _add_continuous(self, state):
        params = state["params"]
        check(len(params.keys() - {"min", "max", "num_grids"}) == 0,
              "At least 1 params is missing for state" + state["name"])

        self.grid_dimensions.append(params["num_grids"])
        _min = params["min"]
        _const = 1 / (params["max"] - _min) * params["num_grids"]
        self.state_converters.append(lambda x: np.floor((np.array(x) - _min) * _const).astype(int))
        self.states[state["name"]] = len(self.states)

    def _add_discrete(self, state):
        params = state["params"]
        check(len(params.keys() - {"min", "max"}) == 0,
              "At least 1 params is missing for state" + state["name"])

        _min, _max = params["min"], params["max"]
        self.grid_dimensions.append(_max - _min + 1)
        self.state_converters.append(lambda x: np.array(x) - _min)
        self.states[state["name"]] = len(self.states)

    def _link_incrementally(self, state):
        # add constraint
        # linked state
        # update no link state
        pass

    def get_grid_value(self, indices):
        check(self.grid is not None, "grid not initialized")
        check(len(indices) == len(self.state_converters), "index dimension doesn't match number of converters")

        converted = []
        for index, converter in zip(indices, self.state_converters):
            converted.append(converter(index))

        return self.grid[np.ix_(*converted)]

    def initialize_grid(self):
        check("action" in self.states, "no action is found")
        check(len(self.grid_dimensions) > 1, "no state is found")

        action_idx = self.states["action"]
        self.states = {st: idx - (idx > action_idx) for st, idx in self.states.items() if st != "action"}
        self.action_converter = self.state_converters[action_idx]
        del self.state_converters[action_idx]
        action_dim = self.grid_dimensions[action_idx]
        del self.grid_dimensions[action_idx]
        self.grid_dimensions.append(action_dim)
        self.grid = np.zeros(self.grid_dimensions)

    def choose(self):
        pass
