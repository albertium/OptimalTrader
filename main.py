
import numpy as np

from _TYPE_OptimalTrader import GridTrader
from _TYPE_StochasticProcess import ExampleOUProcess
from _TYPE_Grid import ActionValueGrid


maze = -np.ones(10)
maze[9] = 100  # last cell is the end point
q_table = ActionValueGrid()
q_table.add_nodes([
    {"name": "location", "type": "discrete", "params": {"min": 0, "max": 9}, "link": "incremental"},
    {"name": "action", "type": "discrete", "params": {"min": -1, "max": 1}}
])
q_table.initialize_grid([0])
for _ in range(100000):
    q_table.current_states = [np.random.randint(0, 10)]
    action = q_table.choose()
    q_table.update([], maze[q_table.current_states[0] + action])

np.set_printoptions(precision=2, suppress=True)
print(q_table.grid)
