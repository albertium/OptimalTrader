
import numpy as np
from _LIB_Core import check, check_components


class BasicGrid:
    """ state only allows single-valued indexing while action allows series indexing """

    def __init__(self, nodes):
        self.state_dimensions = []
        self.state_converters = []

        self._load_config()
        self._add_nodes(nodes)

    def _load_config(self):
        from _CFG_Grid import node_types
        self.node_types = node_types

    def _add_nodes(self, nodes):
        check("action" in nodes, "BasicGrid: no action node found")
        check(len(nodes) > 1, "BasicGrid: no state node found")

        for name, node in nodes.items():
            if name == "action":
                config = self.node_types["discrete"]
                self.action_dimension = config["dimension_func"](node["params"])
                self.action_converter = config["converter"](node["params"])
            else:
                config = self.node_types[node["type"]]
                self.state_dimensions.append(config["dimension_func"](node["params"]))
                self.state_converters.append(config["converter"](node["params"]))

        self.shape = tuple(self.state_dimensions + [self.action_dimension])
        self.table = np.zeros(self.shape)

    def __getitem__(self, item):
        if isinstance(item, list):
            check(len(item) == len(self.state_converters), "BasicGrid: state dimensions mismatch")
            states, action = item, None
        else:
            check(len(item) == 2, "BasicGrid: expected 1 or 2 arguments")
            states, action = item

        converted = tuple(converter(state) for state, converter in zip(states, self.state_converters))
        if action is not None:
            if isinstance(action, slice):
                lb, ub = self.action_converter(action.start), self.action_converter(action.stop)
                return self.table[converted + (slice(lb, ub), )]
            else:
                idx = self.action_converter(action)
                return self.table[converted + (idx,)]
        return self.table[converted]

    def __setitem__(self, key, value):
        if isinstance(key, list):
            check(len(key) == len(self.state_converters), "BasicGrid: state dimensions mismatch")
            states, action = key, None
        else:
            check(len(key) == 2, "BasicGrid: expected 1 or 2 arguments")
            states, action = key

        converted = tuple(converter(state) for state, converter in zip(states, self.state_converters))
        if action is not None:
            if isinstance(action, slice):
                lb, ub = self.action_converter(action.start), self.action_converter(action.stop)
                self.table[converted + (slice(lb, ub),)] = value
            else:
                idx = self.action_converter(action)
                self.table[converted + (idx,)] = value
        else:
            self.table[converted] = value

    def __str__(self):
        np.set_printoptions(precision=2, suppress=True)
        return self.table.__str__()


class TabularGrid:
    def __init__(self):
        # state variables
        self.state_map = {}
        self.state_generators = []
        self.state_movers = {}
        self.current_states = None
        self.next_states = None
        self.no_link_states = set()
        # action variables
        self.action_bound = None
        self.action_constraints_to_add = {}
        self.get_action_bounds = None
        self.current_action = None
        # other variables
        self.link_types = {}
        self.q_table = None

        # training parameters
        self.alpha = 0.001
        self.gamma = 0.999
        self.epsilon = 0.1

        self._load_config()

    def add_specs(self, specs):
        for name, node in specs.items():
            # deal with action
            if name == "action":
                self.action_constraints_to_add[-1] = self.link_types["direct"]["action_constraint"](node["params"])
                continue

            # add map and generators
            self.state_map[name] = len(self.state_map)
            self.state_generators.append(self.node_types[node["type"]]["generator"](node["params"]))
            # add link
            if "link" in node:
                config = self.link_types[node["link"]]
                self.action_constraints_to_add[self.state_map[name]] = config["action_constraint"](node["params"])
                self.state_movers[self.state_map[name]] = config["mover"]
            else:
                self.no_link_states.add(name)

        check(len(self.action_constraints_to_add) > 0, "TabularGrid: no action node is found")
        self.get_action_bounds = self._constrain_all(self.action_constraints_to_add)
        self.q_table = BasicGrid(specs)

    def set_current_states(self, states):
        check(isinstance(states, dict), "TabularGrid: expect \"<class 'dict'>\" instead of %s" % type(states))
        if self.current_states is None:
            self.current_states = [0] * len(self.state_map)
        for name, idx in self.state_map.items():
            if name in states:
                self.current_states[idx] = states[name]
            else:
                self.current_states[idx] = self.state_generators[idx]()

        for state, value in states.items():
            self.current_states[self.state_map[state]] = value

    def choose(self):
        check(self.current_states is not None, "TabularGrid: current state not initialized")
        action_lb, action_ub = self.get_action_bounds(self.current_states)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(action_lb, action_ub + 1)
        else:
            q_values = self.q_table[self.current_states, action_lb: action_ub + 1]
            action = np.argmax(q_values).astype(int) + action_lb
        self._move_states(action)
        self.current_action = action
        return action

    def update(self, new_states, reward):
        """ update q value """
        check(set(new_states.keys()) == self.no_link_states, "TabularGrid: update states don't match")
        for state, value in new_states.items():
            self.next_states[self.state_map[state]] = value

        action_lb, action_ub = self.get_action_bounds(self.next_states)
        max_value = np.max(self.q_table[self.next_states, action_lb: action_ub + 1])
        new_value = \
            (1 - self.alpha) * self.q_table[self.current_states, self.current_action] + \
            self.alpha * (reward + self.gamma * max_value)
        self.q_table[self.current_states, self.current_action] = new_value
        self.current_states = self.next_states

    def get_q_value(self, states, action=None):
        if action is None:
            return self.q_table[states,]
        return self.q_table[states, action]

    def get_average_q_value(self):
        return np.mean(self.q_table.table)

    def _move_states(self, action):
        """ update linked states """
        self.next_states = [0] * len(self.state_map)
        for st, mover in self.state_movers.items():
            self.next_states[st] = mover(action, self.current_states[st])

    def _load_config(self):
        from _CFG_Grid import link_types, node_types
        self.link_types = link_types
        self.node_types = node_types

    @staticmethod
    def _constrain_all(constraints):
        """ to combine all the action constraints """
        if not constraints:
            return None

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
