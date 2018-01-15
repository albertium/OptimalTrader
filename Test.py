
import unittest
import numpy as np

from _TYPE_StochasticProcess import ExampleOUProcess
from _TYPE_OptimalTrader import GridTrader
from _TYPE_Grid import ActionValueGrid


class OptimalTraderTestCase(unittest.TestCase):
    """
    test _TYPE_OptimalTrader.py
    """

    def test_if_integer(self):
        process = ExampleOUProcess()
        trader = GridTrader(process)
        trader.initialize_grid(100, 0, 10, -10, 1000, 10000)
        self.assertIsInstance(trader.process_generate(), np.int64)
        self.assertIsInstance(trader.process_generate(10), np.ndarray)


class ActionValueGridTestCase(unittest.TestCase):
    def test_grid_init(self):
        grid = ActionValueGrid()
        grid.add_nodes({"name": "price1", "type": "continuous", "params": {"min": 0.0, "max": 10.0, "num_grids": 1000}})
        grid.add_nodes({"name": "action", "type": "discrete", "params": {"min": -5, "max": 5}})
        grid.add_nodes([
            {"name": "price2", "type": "continuous", "params": {"min": 0.0, "max": 100.0, "num_grids": 5000}},
            {"name": "position", "type": "discrete", "params": {"min": -10, "max": 10}, "link": "incremental"}
        ])
        grid.initialize_grid([5.5, 55, 10])

        # test grid
        self.assertEqual(grid.grid.shape, (1000, 5000, 21, 11))
        self.assertEqual(len(grid.action_constraints_to_add), 2)
        # test constraints
        self.assertEqual(grid.action_constraints([5.5, 55, 10]), (-5, 0))
        self.assertEqual(grid.action_constraints([5.5, 55, -10]), (0, 5))
        # test movers
        self.assertEqual(len(grid.state_movers), 1)
        self.assertIn(2, grid.state_movers)
        self.assertEqual(grid.state_movers[2](3, -4), -1)
        # test get
        self.assertEqual(grid.get_value([0.0, 0.0, -10]).shape, (11,))
        self.assertEqual(grid.get_value([9.99, 99.999, 10], -3, 1).shape, (5,))
        # test choose
        grid.epsilon = 0
        self.assertEqual(grid.choose(), -5)
        self.assertEqual(grid.current_states[2], 10)
        self.assertEqual(grid.next_states[2], 5)
        # test update
        grid.update([5, 50], 100)
        self.assertEqual(grid.get_value([5.5, 55, 10], -5), 0.1)

    def test_maze(self):
        """ test to see if ActionValueGrid is able to solve 1d maze """
        maze = np.zeros(10)
        maze[9] = 100  # last cell is the end point
        q_table = ActionValueGrid()
        q_table.add_nodes([
            {"name": "location", "type": "discrete", "params": {"min": 0, "max": 9}, "link": "incremental"},
            {"name": "action", "type": "discrete", "params": {"min": -1, "max": 1}}
        ])
        q_table.initialize_grid([0])
        check_bounds = 0
        q_table.alpha, q_table.gamma = 0.1, 0.9
        for _ in range(10000):
            q_table.current_states = [np.random.randint(0, 10)]
            action = q_table.choose()
            pos = q_table.current_states[0] + action
            check_bounds += not 0 <= pos <= 9
            q_table.update([], maze[pos])
        self.assertSequenceEqual(tuple(np.argmax(q_table.grid, 1)), tuple([2] * 9 + [1]))
        self.assertEqual(check_bounds, 0)


if __name__ == "__main__":
    unittest.main()
