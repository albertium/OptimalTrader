
import numpy as np
from numpy import ndarray
import time
import keras
from collections import deque

from typing import List, Union, Tuple
import abc

from _LIB_Core import check, Check, Timer


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
    def choose(self, states: list) -> Union[int, ndarray]:
        pass

    @abc.abstractclassmethod
    def update(self, reward: float, states: list, action: float, new_states: list) -> None:
        pass

    @abc.abstractclassmethod
    def get_average_q_value(self) -> float:
        pass

    @staticmethod
    def get_action_bounds(trade_bounds: tuple, position_bounds: tuple, position: int) -> tuple:
        return max(position_bounds[0] - position, trade_bounds[0]), min(position_bounds[1] - position, trade_bounds[1])


class KerasQAgent(QAgent):
    """
    Contract:
    1. n discrete actions
    2. position will be considered as a natural states, bounds should be taken care by trader
    3. update(R, S, A, S', [A']) - this is the only function that will change the Q table
    4. choose(S) - this will not change neither the Q table nor the states
        a. can provide the available moves
    5. state means single state and states mean multiple states
    """
    def __init__(self, num_states=1, num_actions: int=11, history_len=10000, replay_size=120):
        super().__init__()

        self.num_actions = num_actions

        self.q_map = keras.models.Sequential()
        self.q_map.add(keras.layers.Dense(num_states * 5, input_dim=num_states, activation="relu"))
        self.q_map.add(keras.layers.Dense(self.num_actions))
        self.q_map.compile(optimizer=keras.optimizers.adam(lr=self.alpha), loss="mse")

        self.ema_q_value = 0
        self.reward_history = None
        self.state_history = None
        self.action_history = None
        self.new_state_history = None
        self.available_action_history = []
        self.history_len = history_len
        self.replay_size = replay_size

        # timing
        self.timer = Timer()

        self.check = Check("Keras Q Agent")

    def choose(self, state: ndarray) -> Union[int, ndarray]:
        return self.q_map.predict(state)

    def update(self, reward: float, state: list, action: float, new_state: list) -> None:
        """TODO: change states from list to np.array"""
        # next_values = self.q_map.predict(np.array([new_states + [action] for  1)]))
        # self.ema_q_value = self.gamma * self.ema_q_value + (1 - self.gamma) * np.mean(next_values)
        # max_value = np.max(next_values)
        # new_value = reward + self.gamma * max_value
        # self.q_map.train_on_batch(np.array([states + [action]]), np.array([new_value]))
        pass

    def update_many(self, rewards: ndarray, states: ndarray, actions: ndarray, new_states: ndarray,
                    action_bounds: List[tuple]) -> None:
        self.timer._____________________________________START______________________________________("update_many")
        self.check(states.shape[0] == 1, "states should have only one row")
        next_values = self.q_map.predict(new_states)
        max_values = np.array([np.max(q_values[lb: ub + 1]) for q_values, (lb, ub) in zip(next_values, action_bounds)])
        self.ema_q_value = self.gamma * self.ema_q_value + (1 - self.gamma) * np.mean(max_values)
        new_values = self.gamma * max_values + rewards
        targets = self.q_map.predict(states)
        targets[:, actions] = new_values
        self.q_map.train_on_batch(states, targets)
        self.timer.______________________________________END_______________________________________("update_many")

    def update_with_replay(self, rewards: ndarray, states: ndarray, actions: ndarray,
                           new_states: ndarray, available_actions: List[ndarray]) -> None:
        # start_time = time.time()
        # if self.reward_history is None:
        #     self.reward_history = rewards
        #     self.state_history = states
        #     self.new_state_history = new_states
        #     self.action_history = actions
        # else:
        #     self.reward_history = np.hstack([self.reward_history, rewards])
        #     self.state_history = np.vstack([self.state_history, states])
        #     self.new_state_history = np.vstack([self.new_state_history, new_states])
        #     self.action_history = np.hstack([self.action_history, actions])
        # self.available_action_history += available_actions
        # if len(self.reward_history) > self.history_len:
        #     to_delete = self.state_history.shape[0] - self.history_len
        #     self.reward_history = self.reward_history[to_delete:]
        #     self.state_history = self.state_history[to_delete:, :]
        #     self.action_history = self.action_history[to_delete:]
        #     self.new_state_history = self.new_state_history[to_delete:, :]
        #     del self.available_action_history[:to_delete]
        #
        # end_time = time.time()
        # self.time_replay += end_time - start_time
        #
        # if len(self.reward_history) > self.replay_size * 5:
        #     start_time = time.time()
        #     rand_idx = np.random.choice(range(len(self.reward_history)), self.replay_size, replace=False)
        #     end_time = time.time()
        #     self.time_replay += end_time - start_time
        #
        #     start_time = time.time()
        #     self.update_many(self.reward_history[rand_idx], self.state_history[rand_idx], self.action_history[rand_idx],
        #                      self.new_state_history[rand_idx], [self.available_action_history[idx] for idx in rand_idx])
        #     end_time = time.time()
        #     self.time_many += end_time - start_time
        pass

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
