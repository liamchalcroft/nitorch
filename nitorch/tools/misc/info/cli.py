from nitorch.tools.cli import commands
from .main import info
from .parser import parse, help, ParseError
import sys


def cli(args=None):
    f"""Command-line interface for `niinfo`
    
    {help}
    
    """

    # Exceptions are dealt with here
    try:
        _cli(args)
    except ParseError as e:
        print(help)
        print(f'[ERROR] {str(e)}', file=sys.stderr)
    except Exception as e:
        print(f'[ERROR] {str(e)}', file=sys.stderr)


def _cli(args):
    """Command-line interface for `niinfo` without exception handling"""
    args = args or sys.argv[1:]

    options = parse(args)
    if not options:
        return

    for file in options.files:
        info(file, meta=options.meta, stat=options.stat)


commands['info'] = cli
