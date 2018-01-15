
import numpy as np
import plotly.offline as offline
import plotly.graph_objs as go
import abc


class StochasticProcess:
    def __init__(self, name):
        self.name = name

    @abc.abstractclassmethod
    def _generate(self):
        pass

    @abc.abstractclassmethod
    def generate(self, n=1):
        pass

    @abc.abstractclassmethod
    def plot(self, n=1):
        pass


class UnivariateProcess(StochasticProcess):
    def __init__(self, name):
        super().__init__(name)

    @abc.abstractclassmethod
    def _generate(self):
        pass

    def generate(self, n=1):
        if not isinstance(n, int):
            raise ValueError("n must be an integer")
        if n < 1:
            raise ValueError("n must be at least 1")
        if n == 1:
            return self._generate()
        else:
            result = np.zeros(n)
            for i in range(n):
                result[i] = self._generate()
            return result

    def plot(self, n=1):
        data = self.generate(n)
        trace = go.Scatter(y=data, mode="lines", name=self.name)
        offline.plot([trace])


class ExampleOUProcess(UnivariateProcess):
    def __init__(self):
        super().__init__("Example OU")
        self.pe = 50
        self.half_life = 5
        self.lam = np.log(2) / self.half_life
        self.sigma = 0.1
        self.xt = 0

    def _generate(self):
        self.xt = -self.lam * self.xt + self.sigma * np.random.normal()
        return self.pe * np.exp(self.xt)

