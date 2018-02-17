
import numpy as np
from numpy import ndarray
from typing import List, Union, Tuple
import abc

from _LIB_Core import Check, plot_lines
from _TYPE_StochasticProcess import UnivariateProcess
from _TYPE_Grid import TabularAgent, KerasQAgent


class OptimalTrader:
    def __init__(self, max_trade=5, max_position=10):
        self.max_trade = max_trade
        self.max_position = max_position
        self.specs = {
            "action": {
                "params": {"min": -max_trade, "max": max_trade}
            },
            "position": {
                "type": "discrete",
                "params": {"min": -max_position, "max": max_position},
                "link": "incremental"
            }
        }

        self.process = None
        self.q_table = None
        self.position = 0
        self.feature_callbacks = []
        self.pre_run_lags = 0

    @abc.abstractclassmethod
    def choose(self, price):
        pass

    @abc.abstractclassmethod
    def update(self, price, position, rewards):
        pass

    def trade(self, price: float, position: int=None) -> int:
        if position is not None:
            self.position = position
        return self._trade(price)

    @abc.abstractclassmethod
    def _trade(self, price: float) -> int:
        pass

    @abc.abstractclassmethod
    def get_average_q_value(self) -> float:
        pass

    def initialize_features(self, prices: ndarray) -> None:
        for price in prices:
            for callback in self.feature_callbacks:
                callback(price)

    def _get_action_bounds(self, position) -> tuple:
        lb = max(-self.max_position - position, -self.max_trade)
        ub = min(self.max_position - position, self.max_trade)
        return lb, ub


class UnivariateTabularTrader(OptimalTrader):
    """
    Contract
    1. update(price, reward) - store new features derived from the price, reward can be None. Should be able to handle
                                multiple actions update. It updates features
    2. choose([price]) - uses the stored features to generate all available actions. It updates position and actions
    3. trade(price) - only return one action
    """
    def __init__(self, feature_list: dict, process: UnivariateProcess) -> None:
        super().__init__(max_trade=5, max_position=10)
        self.check = Check("UniTabTrader")
        self._load_config()
        self._add_process(process)  # process should be an object than type since it might contain external data
        self._add_specs(feature_list)

        self.position = 0
        self.actions = None
        self.features = []

    def choose(self, price=None) -> Tuple[int, Union[int, ndarray]]:
        if price is not None:
            self.features = self._generate_features(price)
        self.position = np.random.randint(-self.max_position, self.max_position + 1)
        self.actions = self.q_table.choose([self.position] + self.features, all_actions=True)
        return self.position, self.actions

    def update(self, price: float, rewards: Union[float, ndarray]) -> None:
        new_features = self._generate_features(price)
        for reward, action in zip(rewards, self.actions):
            self.q_table.update(reward, [self.position] + self.features, action, [self.position + action] + new_features)
        self.features = new_features

    def trade(self, price: float, position: int=None) -> int:
        if position is not None:
            self.position = position
        self.features = self._generate_features(price)
        action = self.q_table.choose([self.position] + self.features)
        self.position += action
        return action

    def _trade(self, price: float):
        pass

    def get_average_q_value(self) -> float:
        return self.q_table.get_average_q_value()

    def _add_process(self, process: UnivariateProcess) -> None:
        # only for pre-run
        self.check(isinstance(process, UnivariateProcess), "expects univariate process")
        self.process = process

    def _add_specs(self, feature_list: dict) -> None:
        """
        Example Feature:

        features = {
            "level": {"cells": 1000, "spec": []},

            "lag5": {"cells": 500, "spec": [["apply_lag", 5]]},

            "diff_1_lag_3": {"cells": 500, "spec": [["diff", 1], ["lag", 3]]}
            }
        """
        cells_list = []
        pre_run_lags = 0
        for name, info in feature_list.items():
            cells_list.append(info["cells"])
            spec = info["spec"]

            # apply filters one by one onto the base filter
            total_lag = 0
            func = self.filter_type["level"]
            for filter_name, lag in spec:
                func = self.filter_type[filter_name](func, lag)
                total_lag += lag
            self.feature_callbacks.append(func)
            pre_run_lags = max(pre_run_lags, total_lag + 1)
        self.pre_run_lags = pre_run_lags

        # for grid specification
        bounds = self._get_bounds()  # decide bounds for the feature through simulation

        # add position as the first state
        state_specs = [{
            "type": "discrete",
            "params": {"min": -self.max_position, "max": self.max_position},
            "link": "incremental"
        }]  # type: List[dict]

        # add other features
        for [lb, ub], cells in zip(bounds, cells_list):
            state_specs.append({
                "type": "continuous",
                "params": {"min": lb, "max": ub, "cells": cells}
            })

        self.q_table = TabularAgent(state_specs, {"min": -self.max_trade, "max": self.max_trade})

    def _load_config(self):
        from _CFG_OptimalTrader import filter_type
        self.filter_type = filter_type

    def _generate_features(self, data: float) -> List[float]:
        return [func(data) for func in self.feature_callbacks]

    def _get_bounds(self, buffer: float=0.4, n_epoch: int=10000) -> list:
        data = np.array([self._generate_features(self.process.generate()) for _ in range(n_epoch)]).T
        output = []
        for datum in data:
            _min, _max = np.nanmin(datum), np.nanmax(datum)
            _min, _max = _min / (1 + buffer), _max * (1 + buffer)
            output.append([_min, _max])
        return output


class UnivariateKerasTrader(OptimalTrader):
    def __init__(self):
        super().__init__()
        # we use action directly as position here
        self.max_position = 1
        self.q_map = KerasQAgent(num_states=2, num_actions=self.max_position * 2 + 1)
        self.position = 0
        self.action = None
        self.features = []

    def choose(self, price):
        self.features = [price]
        new_position = self.q_map.choose(np.array([price, self.position])) - self.max_position
        self.action = new_position - self.position
        return self.position, self.action

    def update(self, new_price, reward: ndarray):
        state = np.array(self.features + [self.position])
        self.position += self.action
        new_state = np.array([new_price, self.position])
        self.q_map.update(reward, state, self.position + self.max_position, new_state)

    def get_average_q_value(self):
        return self.q_map.get_average_q_value()

    def _trade(self, price):
        new_position = np.argmax(self.q_map.q_map.predict(np.array([[price, self.position]]))) - self.max_position
        action = new_position - self.position
        self.position = new_position
        return action


