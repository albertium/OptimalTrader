

from _LIB_Core import timeit
import numpy as np
from _TYPE_StochasticProcess import ExampleOUProcess
from _TYPE_OptimalTrader import UnivariateTabularTrader, UnivariateKerasTrader
from _TYPE_Simulator import Simulator

feature_list = {
    "level": {
        "cells": 500,
        "spec": []
    }
}

# process_type = ExampleOUProcess
# trader = UnivariateTabularTrader(feature_list, process_type())
# sim = Simulator(process_type(), trader)
# sim.train(int(100000))
# sim.test(20000)
# print(sim.record)
# sim.record.plot()

process_type = ExampleOUProcess
trader = UnivariateKerasTrader()
sim = Simulator(process_type(), trader)
sim.train(int(100000))
sim.test(20000)
print(sim.record)
sim.record.plot()
trader.q_map.timer.show()

batches = [[price, position]
           for price in [40, 45, 50, 55, 60]
           for position in range(-10, 11)]
batches = np.array(batches)
result = trader.q_map.q_map.predict(batches)
output = np.hstack([batches, result])
np.savetxt("cached.csv", output, delimiter=",")

results = {}
data = trader.q_map.history
for idx, value in data:
    tag = (int(np.digitize(idx[0][0], [40, 45, 50, 55, 60])), idx[0][1])
    if tag in results:
        results[tag].append(np.squeeze(value))
    else:
        results[tag] = [np.squeeze(value)]

output = []
for idx, value in results.items():
    output.append(np.hstack([idx, np.mean(value, axis=0)]))
output = np.vstack(output)
output = output[np.lexsort([output[:, 1], output[:, 0]])]
np.savetxt("cached2.csv", output, delimiter=",")
