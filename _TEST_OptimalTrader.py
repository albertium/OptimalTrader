
"""
Smart HuiHui
"""
import unittest
import math
import numpy as np

from _CFG_OptimalTrader import apply_lag, apply_diff, get_level
from _TYPE_OptimalTrader import UnivariateGridTrader
from _TYPE_StochasticProcess import ExampleOUProcess


class CfgTestCase(unittest.TestCase):
    def test_lag(self):
        lag0 = apply_lag(get_level, 0)
        lag3 = apply_lag(get_level, 3)

        result0 = [lag0(i) for i in range(10)]
        result3 = [lag3(i) for i in range(10)]
        self.assertSequenceEqual(result0, range(10))
        self.assertSequenceEqual(result3[3:], range(7))
        self.assertEqual(np.sum([math.isnan(x) for x in result3]), 3)

    def test_diff(self):
        lag0 = apply_diff(get_level, 0)
        lag3 = apply_diff(get_level, 3)

        result0 = [lag0(i) for i in range(10)]
        result3 = [lag3(i) for i in range(10)]
        self.assertSequenceEqual(result0, [0] * 10)
        self.assertSequenceEqual(result3[3:], [3] * 7)
        self.assertEqual(np.sum([math.isnan(x) for x in result3]), 3)


class OptimalTraderTestCase(unittest.TestCase):
    def test_setup(self):
        trader = UnivariateGridTrader()
        process = ExampleOUProcess()
        trader.add_process(process)
        # add features
        features = {
            "level": {
                "cells": 500,
                "spec": []
            }
        }
        trader.add_features(features)
        self.assertEqual(trader.q_table.q_table.shape, (21, 500, 11))
        self.assertIsInstance(trader.process.update_features()["level"], float)

        # check training outcome
        start_q_value = trader.q_table.get_average_q_value()
        trader.train(30000)
        end_q_value = trader.q_table.get_average_q_value()
        self.assertGreater(end_q_value, start_q_value)


if __name__ == "__main__":
    unittest.main()
