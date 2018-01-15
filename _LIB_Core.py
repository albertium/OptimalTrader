

def check(predicate, msg=""):
    if not predicate:
        raise ValueError(msg)


def check_components(obj, keys):
    for key in keys:
        if key not in obj:
            raise ValueError(key + " not found")
