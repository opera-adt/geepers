# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
from __future__ import annotations

import logging

import tyro

from .workflows import main

logger = logging.getLogger("geepers")
logger.setLevel(logging.INFO)
h = logging.StreamHandler()
f = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
h.setFormatter(f)
logger.addHandler(h)


def cli():  # noqa: D103
    tyro.cli(main)


cli.__doc__ = main.__doc__


if __name__ == "__main__":
    cli()
