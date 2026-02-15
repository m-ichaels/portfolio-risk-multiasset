"""The calendars are execution-ops' (`xops.calendar`: NYSE, LSE, Xetra, Euronext, CME with early closes and one-off
closures; `xops.fi`: CME first-notice and last-trading rules, IMM dates, front-contract selection, roll checks).  This
module finds them: an installed `xops`, then the sibling checkout ../ProjectE, then the copy vendored under
xrisk/_xops for continuous integration (the vendored files are those of ProjectE at the commit in _xops/VERSION)."""
from __future__ import annotations

import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIBLING = os.path.join(os.path.dirname(ROOT), "ProjectE")


def _load():
    try:
        return importlib.import_module("xops.calendar"), importlib.import_module("xops.fi"), "installed"
    except ImportError:
        pass
    if os.path.isdir(os.path.join(SIBLING, "xops")):
        sys.path.insert(0, SIBLING)
        try:
            return importlib.import_module("xops.calendar"), importlib.import_module("xops.fi"), "sibling"
        except ImportError:
            sys.path.pop(0)
    return importlib.import_module("xrisk._xops.calendar"), importlib.import_module("xrisk._xops.fi"), "vendored"


cal, fi, SOURCE = _load()


ALIAS = {"6B": "6E", "6J": "6E"}     # the CME FX contracts share the Euro FX rule (last trade two business days before the third Wednesday)


def futures_dates(root: str, y: int, m: int) -> dict:
    return fi.futures_dates(ALIAS.get(root, root), y, m)
