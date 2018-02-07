
import time
import plotly.offline as offline
import plotly.graph_objs as go
from plotly import tools


class Check:
    def __init__(self, tag=""):
        self.tag = tag

    def __call__(self, *args, **kwargs):
        if not args[0]:
            raise ValueError("[%s]: %s" % (self.tag, args[1]))


def check(predicate, msg=""):
    if not predicate:
        raise ValueError(msg)


def check_components(obj, keys):
    for key in keys:
        if key not in obj:
            raise ValueError(key + " not found")


def timeit(n_iters=1):
    def wrapper(method):
        def apply(*args, **kwargs):
            start_time = time.time()
            result = None
            for _ in range(n_iters):
                result = method(*args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time
            if elapsed < 0.001:
                print("Timed %r:  %.3f ms - %d times" % (method.__name__, elapsed * 1000, n_iters))
            else:
                print("Timed %r:  %.3f s - %d times"% (method.__name__, elapsed, n_iters))
            return result
        return apply
    return wrapper


class Timer:
    def __init__(self):
        self.records = {}
        self.clock = {}

    def _____________________________________START______________________________________(self, tag: str):
        self.clock[tag] = time.time()

    def ______________________________________END_______________________________________(self, tag: str):
        if tag in self.records:
            self.records[tag] += time.time() - self.clock[tag]
        else:
            self.records[tag] = time.time() - self.clock[tag]

    def show(self):
        for tag, lapse in self.records.items():
            print("%-15s %ds" % (tag + ":", lapse))


def plot_lines(data: dict, plot_name: str="untitled", same_plot=True) -> None:
    check(isinstance(data, dict), "plot_lines expects dict as input")
    if same_plot:
        fig = []
        for name, v in data.items():
            fig.append(go.Scatter(y=v, mode="lines", name=name))
    else:
        fig = tools.make_subplots(len(data), 1)
        for name, v in data.items():
            fig.append_trace(go.Scatter(y=v, mode="lines", name=name))
    offline.plot(fig, filename=plot_name + ".html")
