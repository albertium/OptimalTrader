
import numpy as np
from _LIB_Core import check, check_components


class ActionValueGrid:
    def __init__(self):
        self.node_types = {}
        self.link_types = {}
        self.nodes_to_add = []

        self.states = {}
        self.state_dimensions = []
        self.state_converters = []
        self.state_movers = {}
        self.no_link_states = []
        self.current_states = None
        self.next_states = None

        self.action_dimension = None
        self.action_converter = None
        self.action_constraints_to_add = {}
        self.action_constraints = None
        self.current_action = None

        self.grid = None

        # training parameters
        self.alpha = 0.001
        self.gamma = 0.999
        self.epsilon = 0.1

        self.load_config()

    def load_config(self):
        from _CFG_Grid import node_types, link_types
        self.node_types = node_types
        self.link_types = link_types

    def add_nodes(self, nodes):
        if isinstance(nodes, dict):
            nodes = [nodes]

        for node in nodes:
            check("name" in node, "Please specify node name")
            check("type" in node, "Please specify type for " + node["name"])
            check("params" in node, "Please provide params for " + node["name"])
            check(node["type"] in self.node_types, "unknown node type for " + node["name"])
            if "link" in node:
                check(node["link"] in self.link_types, "unknown link type for " + node["name"])
            self.nodes_to_add.append(node)

    def _add_nodes(self):
        """ add states and action based on nodes_to_add """
        for node in self.nodes_to_add:
            config = self.node_types[node["type"]]
            params = node["params"]
            check_components(params, config["required_params"])

            if node["name"] == "action":
                self.action_dimension = config["dimension_func"](params)
                self.action_converter = config["converter"](params)
                self.action_constraints_to_add[-1] = self.link_types["direct"]["action_constraint"](params)
            else:
                self.state_dimensions.append(config["dimension_func"](params))
                self.state_converters.append(config["converter"](params))
                self.states[node["name"]] = len(self.states)

            if "link" in node:
                config = self.link_types[node["link"]]
                self.action_constraints_to_add[self.states[node["name"]]] = config["action_constraint"](params)
                self.state_movers[self.states[node["name"]]] = config["mover"]
            elif node["name"] != "action":
                self.no_link_states.append(self.states[node["name"]])

        if self.action_constraints_to_add:
            self.action_constraints = self._constrain_all(self.action_constraints_to_add)

    @staticmethod
    def _constrain_all(constraints):
        """ to combine all the action constraints """
        def wrapper(current_states):
            _min, _max = -np.inf, np.inf
            for st, constraint in constraints.items():
                if st >= 0:
                    lb, ub = constraint(current_states[st])
                else:  # for acton constraint
                    lb, ub = constraint(0)
                _min, _max = max(_min, lb), min(_max, ub)
            return _min, _max
        return wrapper

    def initialize_grid(self, initial_states):
        if not isinstance(initial_states, list):
            initial_states = [initial_states]
        self._add_nodes()
        check(self.action_dimension is not None, "No action is found")
        check(len(self.state_dimensions) > 0, "No state is found")
        dim = self.state_dimensions + [self.action_dimension]
        # self.grid = np.random.rand(*dim)
        self.grid = np.zeros(dim)

        self.check_states(initial_states)
        self.current_states = np.array(initial_states)

    def check_states(self, states):
        if len(states) != len(self.state_dimensions):
            raise ValueError("state dimensions don't match")
        for st, converter, dim in zip(states, self.state_converters, self.state_dimensions):
            if not (0 <= converter(st) < dim):
                raise ValueError("state value exceeds boundary")

    def get_value(self, states, action_lb=None, action_ub=None):
        check(self.grid is not None, "grid not initialized")
        check(len(states) == len(self.state_converters), "index dimension doesn't match number of converters")

        converted = [converter(state) for state, converter in zip(states, self.state_converters)]
        if action_lb is not None:
            if action_ub is not None:
                lb, ub = self.action_converter([action_lb, action_ub])
                return self.grid[tuple(converted) + (slice(lb, ub + 1), )]
            else:
                lb = self.action_converter(action_lb)
                return self.grid[tuple(converted) + (lb,)]
        return self.grid[tuple(converted)]

    def set_value(self, states, action, value):
        """ assign new value to a cell """
        check(self.grid is not None, "grid not initialized")
        check(len(states) == len(self.state_converters), "index dimension doesn't match number of converters")

        converted = [converter(state) for state, converter in zip(states, self.state_converters)]
        self.grid[tuple(converted) + (self.action_converter(action), )] = value

    def move_states(self, action):
        """ update linked states """
        self.next_states = [0] * len(self.current_states)
        for st, mover in self.state_movers.items():
            self.next_states[st] = mover(action, self.current_states[st])

    def choose(self):
        action_lb, action_ub = self.action_constraints(self.current_states)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(action_lb, action_ub + 1)
        else:
            values = self.get_value(self.current_states, action_lb, action_ub)
            action = np.argmax(values) + action_lb
        self.move_states(action)
        self.current_action = action
        return action

    def update(self, new_states, reward):
        """ update q value """
        check(len(new_states) == len(self.no_link_states), "states dimension doesn't match")

        for idx, val in enumerate(new_states):
            self.next_states[idx] = val
        action_lb, action_ub = self.action_constraints(self.next_states)
        max_value = np.max(self.get_value(self.next_states, action_lb, action_ub))
        new_value = (1 - self.alpha) * self.get_value(self.current_states, self.current_action) + \
            self.alpha * (reward + self.gamma * max_value)
        # new_value = reward + self.gamma * max_value
        self.set_value(self.current_states, self.current_action, new_value)
        self.current_states = self.next_states
