"""Fixture exercising definitions, imports, and calls for the structural index."""

import os
from collections import OrderedDict
import json as j
from . import sibling


def helper(x):
    return x + 1


def caller(x):
    return helper(x)


class Service:
    def run(self):
        return self.prepare()

    def prepare(self):
        return "ready"

    class Config:
        def describe(self):
            return "config"


class App:
    @app.route("/x")
    def handler(self):
        return "ok"


caller(1)
