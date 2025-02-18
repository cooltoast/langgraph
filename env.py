import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


def set_keys():
    _set_env("ANTHROPIC_API_KEY")
    _set_env("TAVILY_API_KEY")
