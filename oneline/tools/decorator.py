from time import time


def time_monitor(name=None, precision: int=None):
    """
    A decorator for recording the execution time of functions.

    :param name: the name of function that would be shown in the result
    :param precision: the precision of final time
    :return:
    """
    def time_counter(function):
        def wrapper():
            start = time()
            function()
            end = time()
            if precision:
                final = round(end - start, precision)
            else:
                final = end - start
            print(f"{ name if name else 'Function' } execution time: { final }")
        return wrapper
    return time_counter
