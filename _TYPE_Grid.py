
import numpy as np
from numpy import ndarray
import keras
from collections import deque

from typing import List, Union
import abc

from _LIB_Core import check, check_components


class BasicGrid:
    """
    Contract:
    1. provide bracket access to the grid [S, A], where A can be a range and S can only be single values
    2. provide the conversion needed to bridge raw S and internal S
    3. this is boundary safe (floor or cap out-of-bound indexing)
    4. this won't implement any constraints on the states or action
    """

    def __init__(self, states: list, action: dict):
        """
        :param list states: dimensions will be created according to the ordering of nodes
        :param dict action: specification for action node
        """
        self.state_dimensions = []
        self.state_converters = []
        self.state_bounds = []

        self._load_config()
        self._add_nodes(states, action)

    def _load_config(self):
        from _CFG_Grid import node_types
        self.node_types = node_types

    def _add_nodes(self, state_specs: list, action_spec: dict) -> None:
        """
        :param list state_specs: [params, params, ...]
        :param dict action_spec: specification for action node
        :return: None
        """
        # add states
        for node in state_specs:
            config = self.node_types[node["type"]]
            self.state_dimensions.append(config["dimension_func"](node["params"]))
            self.state_converters.append(config["converter"](node["params"]))
            self.state_bounds.append([node["params"]["min"], node["params"]["max"]])

        # add action
        config = self.node_types["discrete"]
        self.action_dimension = config["dimension_func"](action_spec)
        self.action_converter = config["converter"](action_spec)

        self.shape = tuple(self.state_dimensions + [self.action_dimension])
        self.table = np.zeros(self.shape)

    def __getitem__(self, item) -> np.ndarray:
        """
        :param item: can use either [S, A] or [S]. Action can either be a single value of slice
        """
        if isinstance(item, list):
            states, action = item, None
        else:
            check(len(item) == 2, "BasicGrid: expected 1 or 2 arguments")
            states, action = item

        converted = tuple(converter(min(max(state, bound[0]), bound[1]))
                          for state, converter, bound
                          in zip(states, self.state_converters, self.state_bounds))
        if action is not None:
            if isinstance(action, slice):
                lb, ub = self.action_converter(action.start), self.action_converter(action.stop)
                return self.table[converted + (slice(lb, ub), )]
            else:
                idx = self.action_converter(action)
                return self.table[converted + (idx,)]
        return self.table[converted]

    def __setitem__(self, key, value) -> None:
        if isinstance(key, list):
            states, action = key, None
        else:
            check(len(key) == 2, "BasicGrid: expected 1 or 2 arguments")
            states, action = key

        converted = tuple(converter(min(max(state, bound[0]), bound[1]))
                          for state, converter, bound
                          in zip(states, self.state_converters, self.state_bounds))
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


class QAgent:
    """
    Contract:
    1. update(R, S, A, S', [A']) - this is the only function that will change the Q table
    2. choose(S) - this will not change neither the Q table nor the states
        a. can provide the available moves
    """
    def __init__(self):
        # training parameters
        self.alpha = 0.001
        self.gamma = 0.999
        self.epsilon = 0.1

    @abc.abstractclassmethod
    def choose(self, states: list, all_actions: bool) -> Union[int, ndarray]:
        pass

    @abc.abstractclassmethod
    def update(self, reward: float, states: list, action: float, new_states: list) -> None:
        pass

    @abc.abstractclassmethod
    def get_average_q_value(self) -> float:
        pass

    @staticmethod
    def get_action_bounds(max_trade, position_bounds, position):
        return max(position_bounds[0] - position, -max_trade), min(position_bounds[1] - position, max_trade)


class KerasQAgent(QAgent):
    """
    Contract:
    1. discrete actions with bounds
    2. position with bounds
    3. update(R, S, A, S', [A']) - this is the only function that will change the Q table
    4. choose(S) - this will not change neither the Q table nor the states
        a. can provide the available moves
    """
    def __init__(self, num_features=0, max_trades: int=5, position_bounds: list=None):
        super().__init__()

        self.max_trades = max_trades
        if position_bounds is None:
            self.position_bounds = [-10, 10]
        else:
            self.position_bounds = position_bounds

        self.q_map = keras.models.Sequential()
        self.q_map.add(keras.layers.Dense(num_features+2, input_dim=num_features+2, activation="relu"))
        self.q_map.add(keras.layers.Dense(1))
        self.q_map.compile(optimizer=keras.optimizers.adam(lr=self.alpha), loss="mse")

        self.ema_q_value = 0

    def choose(self, states: list, all_actions: bool=False) -> Union[int, ndarray]:
        # last state is position
        action_lb, action_ub = self.get_action_bounds(self.max_trades, self.position_bounds, states[-1])
        if all_actions:
            return np.arange(action_lb, action_ub + 1)
        if np.random.rand() < self.epsilon:
            return np.random.randint(action_lb, action_ub + 1)
        actions = np.arange(action_lb, action_ub + 1)
        q_values = self.q_map.predict(np.array([states + [action] for action in actions]))
        return actions[np.argmax(q_values)]

    def update(self, reward: float, states: list, action: float, new_states: list) -> None:
        """TODO: change states from list to np.array"""
        action_lb, action_ub = self.get_action_bounds(self.max_trades, self.position_bounds, new_states[-1])
        next_values = self.q_map.predict(np.array([new_states + [action] for action in range(action_lb, action_ub + 1)]))
        self.ema_q_value = self.gamma * self.ema_q_value + (1 - self.gamma) * np.mean(next_values)
        max_value = np.max(next_values)
        new_value = reward + self.gamma * max_value
        self.q_map.train_on_batch(np.array([states + [action]]), np.array([new_value]))

    def get_average_q_value(self):
        return self.ema_q_value


class TabularAgent:
    """
    Contract:
    1. allows different policies on action selection and value update
    2. update(R, S, A, S', [A']) - this is the only function that will change the Q table
    3. choose(S) - this will not change neither the Q table nor the states
        a. will provide the available moves
    """
    def __init__(self, state_specs: List[dict], action_spec: dict):
        # action variables
        self.action_constraints_to_add = []
        self.get_action_bounds = None
        # other variables
        self.q_table = None

        # training parameters
        self.alpha = 0.001
        self.gamma = 0.999
        self.epsilon = 0.1

        self._load_config()
        self._add_specs(state_specs, action_spec)

    def choose(self, states: list, all_actions: bool=False) -> Union[int, ndarray]:
        action_lb, action_ub = self.get_action_bounds(states)  # this is created by lumping all the constraints lambda
        if all_actions:
            return np.linspace(action_lb, action_ub, action_ub - action_lb + 1)

        if np.random.rand() < self.epsilon:
            action = np.random.randint(action_lb, action_ub + 1)
        else:
            q_values = self.q_table[states, action_lb: action_ub + 1]
            action = (np.argmax(q_values).astype(int) + action_lb).item()
        return action

    def update(self, reward: float, states: list, action: float, new_states: list) -> None:
        """ update q value """
        action_lb, action_ub = self.get_action_bounds(new_states)
        max_value = np.max(self.q_table[new_states, action_lb: action_ub + 1])
        new_value = (1 - self.alpha) * self.q_table[states, action] + self.alpha * (reward + self.gamma * max_value)
        self.q_table[states, action] = new_value

    def get_average_q_value(self):
        return np.mean(self.q_table.table)

    def _load_config(self):
        from _CFG_Grid import link_types, node_types
        self.link_types = link_types
        self.node_types = node_types

    def _add_specs(self, state_specs: list, action_spec: dict) -> None:
        """
        :param list state_specs: [node, node, ...]
        :param dict action_spec: params
        :return:
        """
        # default action constraint to make sure available action is within action bounds
        self.action_constraints_to_add.append(self.link_types["direct"]["action_constraint"](action_spec))

        for node in state_specs:
            if "link" in node:
                config = self.link_types[node["link"]]
                self.action_constraints_to_add.append(config["action_constraint"](node["params"]))
            else:
                self.action_constraints_to_add.append(None)

        self.get_action_bounds = self._constrain_all(self.action_constraints_to_add)
        self.q_table = BasicGrid(state_specs, action_spec)

    @staticmethod
    def _constrain_all(constraints: list) -> callable:
        """ to combine all the action constraints """
        if not constraints:
            return None

        def wrapper(states: list) -> tuple:
            _min, _max = constraints[0](0)
            for state, constraint in zip(states, constraints[1:]):
                if constraint is None:
                    continue
                lb, ub = constraint(state)
                _min, _max = max(_min, lb), min(_max, ub)
            return _min, _max
        return wrapper
