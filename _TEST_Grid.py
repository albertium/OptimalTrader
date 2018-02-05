
import unittest

import numpy as np

from _TYPE_Grid import BasicGrid, TabularAgent, KerasQAgent


class BasicGridTestCase(unittest.TestCase):
    def runTest(self):
        self.test_getitem()

    def test_getitem(self):
        nodes = [
            {
                "type": "continuous",
                "params": {"min": 0.0, "max": 10.0, "cells": 5}
            },
            {
                "type": "discrete",
                "params": {"min": 0, "max": 5}
            }
        ]
        action_spec = {"min": -1, "max": 1}
        grid = BasicGrid(nodes, action_spec)
        self.assertEqual(grid.table.shape, (5, 6, 3))
        grid[[5.5, 3]] = [1, 2, 3]
        self.assertEqual(grid[[5.5, 3]].shape, (3,))
        self.assertEqual(tuple(grid[[5.5, 3]]), tuple([1, 2, 3]))
        self.assertEqual(grid[[5.5, 3], 0:1], 2)
        grid[[2, 4], 1] = 100
        self.assertEqual(grid[[2, 4], 1], 100)
        self.assertEqual(np.sum(grid.table), 106)


class TabularAgentTestCase(unittest.TestCase):
    def runTest(self):
        self.test_maze()

    def test_maze(self):
        states = [
            {
                "type": "discrete",
                "params": {"min": 0, "max": 5},
                "link": "incremental"
            }
        ]
        action_spec = {"min": -1, "max": 1}
        q_table = TabularAgent(states, action_spec)

        maze = np.zeros(6)
        maze[5] = 10

        # test constraints
        self.assertEqual(q_table.get_action_bounds([3]), (-1, 1))
        self.assertEqual(q_table.get_action_bounds([0]), (0, 1))
        self.assertEqual(q_table.get_action_bounds([5]), (-1, 0))

        # test choose
        self.assertTrue(np.all(q_table.choose([0], all_actions=True) == [0, 1]))
        self.assertTrue(np.all(q_table.choose([5], all_actions=True) == [-1, 0]))
        self.assertTrue(np.all(q_table.choose([3], all_actions=True) == [-1, 0, 1]))

        # test maze
        q_table.alpha, q_table.gamma = 0.1, 0.9
        for _ in range(5000):
            position = np.random.randint(0, 6)
            action = q_table.choose([position])
            q_table.update(maze[position + action], [position], action, [position + action])
            position += action

        self.assertSequenceEqual(tuple(np.argmax(q_table.q_table.table, 1)), tuple([2] * 5 + [1]))


class KerasQAgentTestCase(unittest.TestCase):
    def runTest(self):
        self.test_maze()

    def test_maze(self):
        maze = np.zeros(6)
        maze[5] = 10
        agent = KerasQAgent(max_trades=1, position_bounds=[0, 5])
        for _ in range(10000):
            position = np.random.randint(0, 6)
            action = agent.choose([position])
            agent.update(maze[position + action], [position], action, [position + action])

        batch = np.array([[pos, act] for pos in range(6) for act in range(-1, 2)])
        q_values = agent.q_map.predict(batch).reshape([6, 3])
        self.assertSequenceEqual(tuple(np.argmax(q_values, 1)), tuple([2] * 6))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(BasicGridTestCase())
    suite.addTest(TabularAgentTestCase())
    suite.addTest(KerasQAgentTestCase())
    runner = unittest.TextTestRunner()
    runner.run(suite)
    # unittest.main()
