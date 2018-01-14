
import unittest
import numpy as np

from StochasticProcess import ExampleOUProcess
from OptimalTrader import GridTrader
from Grid import ActionValueGrid


class OptimalTraderTestCase(unittest.TestCase):
    """
    test OptimalTrader.py
    """

    def test_if_integer(self):
        process = ExampleOUProcess()
        trader = GridTrader(process)
        trader.initialize_grid(100, 0, 10, -10, 1000, 10000)
        self.assertTrue(isinstance(trader.process_generate(), np.int64))

    def test_if_array(self):
        process = ExampleOUProcess()
        trader = GridTrader(process)
        trader.initialize_grid(100, 0, 10, -10, 1000, 10000)
        self.assertTrue(isinstance(trader.process_generate(10), np.ndarray))


class ActionValueGridTestCase(unittest.TestCase):
    def test_grid_init(self):
        grid = ActionValueGrid()
        grid.add_states({"name": "price1", "type": "continuous",
                         "params": {"min": 0.0, "max": 10.0, "num_grids": 1000}})
        grid.add_action({"type": "discrete", "params": {"min": -20, "max": 20}})
        grid.add_states([
            {"name": "price2", "type": "continuous", "params": {"min": 0.0, "max": 100.0, "num_grids": 5000}},
            {"name": "position", "type": "discrete", "params": {"min": -10, "max": 10}}
        ])
        grid.initialize_grid()

        self.assertEqual(grid.grid.shape, (1000, 5000, 21, 41))
        self.assertEqual(grid.get_grid_value([[0.0, 9.999], [0.1, 99.9], [-10, 10]]).shape, (2, 2, 2, 41))


if __name__ == "__main__":
    unittest.main()
