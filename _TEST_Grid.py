
import unittest

import numpy as np

from _TYPE_Grid import BasicGrid, TabularGrid


# class OptimalTraderTestCase(unittest.TestCase):
#     """
#     test _TYPE_OptimalTrader.py
#     """
#
#     def test_if_integer(self):
#         process = ExampleOUProcess()
#         trader = GridTrader(process)
#         trader.initialize_grid(100, 0, 10, -10, 1000, 10000)
#         self.assertIsInstance(trader.process_generate(), np.int64)
#         self.assertIsInstance(trader.process_generate(10), np.ndarray)


class BasicGridTestCase(unittest.TestCase):
    def test_getitem(self):
        nodes = {
            "action": {
                "params": {"min": -1, "max": 1}
            },
            "maze_x": {
                "type": "continuous",
                "params": {"min": 0.0, "max": 10.0, "num_cells": 5}
            },
            "maze_y": {
                "type": "discrete",
                "params": {"min": 0, "max": 5}
            }
        }
        grid = BasicGrid(nodes)
        self.assertEqual(grid.table.shape, (5, 6, 3))
        grid[[5.5, 3]] = [1, 2, 3]
        self.assertEqual(grid[[5.5, 3]].shape, (3,))
        self.assertEqual(tuple(grid[[5.5, 3]]), tuple([1, 2, 3]))
        self.assertEqual(grid[[5.5, 3], 0:1], 2)
        grid[[2, 4], 1] = 100
        self.assertEqual(grid[[2, 4], 1], 100)
        self.assertEqual(np.sum(grid.table), 106)


class TabularGridTestCase(unittest.TestCase):
    def test_maze(self):
        specs = {
            "action": {
                "params": {"min": -1, "max": 1}
            },
            "maze": {
                "type": "discrete",
                "params": {"min": 0, "max": 5},
                "link": "incremental"
            }
        }
        q_table = TabularGrid()
        q_table.add_specs(specs)
        q_table.set_current_states({
            "maze": 0
        })

        maze = np.zeros(6)
        maze[5] = 10

        # test constraints
        self.assertEqual(q_table.get_action_bounds([3]), (-1, 1))
        self.assertEqual(q_table.get_action_bounds([0]), (0, 1))
        self.assertEqual(q_table.get_action_bounds([5]), (-1, 0))

        # test choose
        actions = []
        for _ in range(100):
            actions.append(q_table.choose())
        self.assertLessEqual(np.max(actions), 1)
        self.assertGreaterEqual(np.min(actions), -1)

        q_table.alpha, q_table.gamma = 0.1, 0.9
        check_pos = 0
        for _ in range(5000):
            q_table.set_current_states({})
            action = q_table.choose()
            pos = q_table.current_states[0] + action
            q_table.update({}, maze[pos])
            check_pos += q_table.current_states[0] != pos
        self.assertSequenceEqual(tuple(np.argmax(q_table.q_table.table, 1)), tuple([2] * 5 + [1]))
        # check movers
        self.assertFalse(check_pos)


if __name__ == "__main__":
    unittest.main()
