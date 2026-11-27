"""Compat entrypoint for `python dev.py start`."""

from api.main import app, main

if __name__ == "__main__":
    main()
