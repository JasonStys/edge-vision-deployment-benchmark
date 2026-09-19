"""File: Makes ``python -m edgevision`` invoke the documented benchmark pipeline CLI.

Functions: main is imported and executed. Variables: no module-owned variables are declared; exact
entry-point lines are generated in ``docs/CODE_INDEX.md``.
"""

from edgevision.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
