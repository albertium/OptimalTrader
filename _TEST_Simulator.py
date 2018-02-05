
import unittest
import numpy as np

from _TYPE_StochasticProcess import UnivariateProcess, ExampleOUProcess
from _TYPE_OptimalTrader import OptimalTrader, UnivariateTabularTrader
from _TYPE_Simulator import Simulator


class LinearProcess(UnivariateProcess):
    def __init__(self):
        super().__init__("linear")
        self.counter = 0

    def _generate(self):
        self.counter += 1
        return self.counter


class RandomTrader(OptimalTrader):
    def __init__(self):
        super().__init__()
        self.success = True
        self.epsilon = 1

    def choose(self, price):
        return [0, np.array([-1, 1])]

    def update(self, price, rewards):
        self.success &= np.all(rewards == np.array([-1.00005, 0.99995]))

    def trade(self, price):
        if np.random.rand() < self.epsilon:
            return 1
        return -1


class SimulatorTestCase(unittest.TestCase):
    def test_simulator(self):
        process = LinearProcess()

        # deterministic trading
        trader = RandomTrader()
        simulator = Simulator(process, trader, {"commission": 0, "spread": 0})
        simulator.train(3)
        self.assertTrue(trader.success)
        simulator.test(10)
        record = simulator.record.record
        stats = simulator.record.get_performance()
        self.assertTrue(np.all(record == np.ones(10)))
        self.assertEqual(stats["gross_profit"], 10)
        self.assertEqual(stats["percent_win"], 1)
        self.assertEqual(stats["max_drawdown"], 0)

        # check drawdown
        trader.epsilon = 0
        simulator.test(10)
        stats = simulator.record.get_performance()
        self.assertEqual(stats["max_drawdown"], -10)

        # randomize trading
        trader.epsilon = 0.5
        simulator.test(10000)
        # print(simulator.record)
        # simulator.record.plot()
        self.assertLessEqual(abs(simulator.record.percent_win - .5), 0.03)

    def test_OU_performance(self):
        features = {
            "level": {
                "cells": 500,
                "spec": []
            }
        }
        trader = UnivariateTabularTrader(features, ExampleOUProcess())
        simulator = Simulator(ExampleOUProcess(), trader, {})
        simulator.test(10000)
        self.assertLess(simulator.record.total_pnl, 0)
        self.assertLess(simulator.record.percent_win, .3)

        simulator.train(100000)
        simulator.test(10000)
        self.assertGreater(simulator.record.total_pnl, 0)
        self.assertGreater(simulator.record.percent_win, .5)
        print(simulator.record)


if __name__ == "__main__":
    unittest.main()
