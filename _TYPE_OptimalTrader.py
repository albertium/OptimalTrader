
import numpy as np
import abc

from _LIB_Core import Check, plot_lines
from _TYPE_StochasticProcess import UnivariateProcess
from _TYPE_Grid import TabularGrid


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

    @abc.abstractclassmethod
    def choose(self, price):
        pass

    @abc.abstractclassmethod
    def update(self, rewards):
        pass

    @abc.abstractclassmethod
    def trade(self, price):
        pass


class UnivariateGridTrader(OptimalTrader):
    def __init__(self, max_trade=5, max_position=10):
        super().__init__(max_trade, max_position)
        self.check = Check("UniGridTrader")
        self._load_config()

    def add_process(self, process):
        self.check(isinstance(process, UnivariateProcess), "expects univariate process")
        self.process = process

    def add_features(self, features):
        """
        Example Feature:

        features = {
            "level": {"cells": 1000, "spec": []},

            "lag5": {"cells": 500, "spec": [["apply_lag", 5]]},

            "diff_1_lag_3": {"cells": 500, "spec": [["apply_diff", 1], ["apply_lag", 3]]}
            }
        """
        self.check(isinstance(features, dict), "expects \"<class 'dict'>\" instead of %s" % type(features))
        self.check(self.process is not None, "please add process first")
        for name, spec_info in features.items():
            self.check("cells" in spec_info, "please specify number of cells for feature " + name)
            cells = spec_info["cells"]
            self.check("spec" in spec_info, "please specify spec for feature " + name)
            spec = spec_info["spec"]

            # apply filters one by one onto the base filter
            func = self.filter_base
            for filter_name, lag in spec:
                func = self.filter_type[filter_name](func, lag)
            self.process.register_feature(name, func)  # just for feature not for grid

            # for grid specification
            params = self._get_bounds(func, cells)  # decide bounds for the feature through simulation
            self.specs[name] = {
                "type": "continuous",
                "params": params
            }

        self.q_table = TabularGrid()
        self.q_table.add_specs(self.specs)

    def train(self, n_epochs=100000):
        # pre-run to get rid of NaNs
        print("Training ... [0%]", end="")
        while np.isnan(self.process.update_features(return_list=True)).any():
            pass

        # main training loop
        curr_state = self.process.update_features()
        kappa = 0.0001
        tick_size = 0.2
        q_values = []
        for epoch in range(n_epochs):
            # randomize position
            self.q_table.set_current_states(curr_state)
            action = self.q_table.choose()
            position = self._get_position()
            next_state = self.process.update_features()

            # calculate reward
            dw = position * (next_state["level"] / curr_state["level"] - 1)
            # dw -= tick_size * action + tick_size / self.lot_size * action * action
            reward = dw - 0.5 * kappa * dw * dw

            # update q value
            self.q_table.update(next_state, reward)
            curr_state = next_state

            # monitor
            if (epoch + 1) % 1000 == 0:
                q_values.append(self.q_table.get_average_q_value())
                print("\rTraining ... [%d%%]" % (epoch / n_epochs * 100), end="")

        print()
        plot_lines({"average q": q_values})

    def test(self, n_epochs=10000):
        current_price = self.process.update_features()
        current_price["position"] = 0  # set initial position to 0
        position = 0
        cash = 0
        equity = 0
        equity_curve = []
        for _ in range(n_epochs):
            self.q_table.set_current_states(current_price)  # input current price
            action = self.q_table.choose()
            position += action
            new_price = self.process.update_features()  # get new price
            cash -= action * self.lot_size * current_price["level"]
            equity += position * self.lot_size * new_price["level"]
            equity_curve.append(cash + equity)
            current_price = new_price

        equity_curve = np.array(equity_curve)
        plot_lines({"Equity": equity_curve})
        ret = equity_curve[1:] / equity_curve[:-1]
        print("Sharpe Ratio: %.2f" % (float(np.mean(ret) / np.std(ret))))

    def _load_config(self):
        from _CFG_OptimalTrader import filter_type, get_level
        self.filter_type = filter_type
        self.filter_base = get_level

    def _get_bounds(self, func, num_cells=1000, buffer=0.2, n_epoch=1000):
        data = [func(self.process.generate()) for _ in range(n_epoch)]
        _min, _max = np.nanmin(data), np.nanmax(data)
        _min, _max = _min / (1 + buffer), _max * (1 + buffer)
        # _step = int((_max - _min) / step)
        return {"min": _min, "max": _max, "num_cells": num_cells}

    def _get_position(self):
        return self.q_table.next_states[self.q_table.state_map["position"]]
