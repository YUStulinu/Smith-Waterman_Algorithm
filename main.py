"""Entry point of the command line program.

Run ``python main.py --help`` for the options, or ``python main.py`` without
arguments for the interactive mode. The graphical interface is in ``gui.py``.
"""

from smith_waterman.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
