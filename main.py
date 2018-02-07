

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
