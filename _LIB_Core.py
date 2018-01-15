
import time


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
