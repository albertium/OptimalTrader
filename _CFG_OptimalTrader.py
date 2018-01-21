
from collections import deque


# ----- filters ------
def get_level(x):
    return x


def apply_lag(func, lag):
    queue = deque([float("nan")] * lag)

    def apply(x):
        data = func(x)
        queue.append(data)
        return queue.popleft()
    return apply


def apply_diff(func, lag):
    queue = deque([float("nan")] * lag)

    def apply(x):
        data = func(x)
        queue.append(data)
        return data - queue.popleft()
    return apply


def apply_ret(func, lag):
    queue = deque([float("nan")] * lag)

    def apply(x):
        data = func(x)
        queue.append(data)
        return data / queue.popleft() - 1
    return apply


filter_type = {
    "lag": apply_lag,
    "diff": apply_diff,
    "ret": apply_ret
}
