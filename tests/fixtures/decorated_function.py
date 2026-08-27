"""Fixture: a decorated top-level function and a decorated method."""

import functools


def simple_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@simple_decorator
@functools.lru_cache(maxsize=None)
def decorated_top_level(n):
    """A decorated top-level function."""
    return n * n


class WithDecoratedMethod:
    @staticmethod
    @simple_decorator
    def decorated_method(x):
        return x + 1
