

def check(predicate, msg=""):
    if not predicate:
        raise ValueError(msg)
