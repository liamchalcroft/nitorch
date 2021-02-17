from .main import orient
from .parser import parse, help, ParseError
import sys


def cli(args=None):
    f"""Command-line interface for `niorient`
    
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
    """Command-line interface for `niorient` without exception handling"""
    args = args or sys.argv[1:]

    options = parse(args)
    if not options:
        return

    for fname in options.files:
        orient(fname, layout=options.layout, voxel_size=options.voxel_size,
               center=options.center, like=options.like, output=options.output)