
"""
Smart HuiHui
"""
import unittest
import math
import numpy as np

from _CFG_OptimalTrader import apply_lag, apply_diff, get_level
from _TYPE_OptimalTrader import UnivariateTabularTrader
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
        features = {
            "level": {
                "cells": 500,
                "spec": []
            }
        }
        trader = UnivariateTabularTrader(features, ExampleOUProcess())
        self.assertEqual(trader.q_table.q_table.shape, (21, 500, 11))
        self.assertEqual(trader._generate_features(123.4), [123.4])
        self.assertEqual(trader.pre_run_lags, 1)

        # check choose multiples
        position, actions = trader.choose(50.5)  # mean of Example OU is 50
        self.assertEqual(trader.features, [50.5])
        self.assertTrue(-10 <= position <= 10)
        self.assertGreaterEqual(len(actions), 6)
        self.assertEqual(position, trader.position)
        self.assertTrue((actions == trader.actions).all())

        # check update multiples
        trader.update(56, np.ones_like(actions))
        self.assertEqual(trader.features, [56])
        self.assertEqual(np.sum(trader.q_table.q_table.table) / trader.q_table.alpha, len(actions))

        # check trade
        action = trader.trade(45, 0)
        self.assertIsInstance(action, int)
        self.assertEqual(trader.features, [45])
        self.assertEqual(trader.position, action)


if __name__ == "__main__":
    unittest.main()
