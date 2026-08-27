"""Fixture: a class containing a nested class and methods at both levels."""


class Outer:
    """Outer class docstring."""

    class_attr = 1

    def outer_method(self, value):
        return value + self.class_attr

    class Inner:
        """Inner nested class."""

        def inner_method(self):
            return "inner"

        def another_inner_method(self, x, y):
            total = x + y
            return total

    def after_nested(self):
        return Outer.Inner()
