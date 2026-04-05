from IPython import start_ipython
import argparse,os

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument('-r', type=int, nargs='?', const=-1, default=None)
_p.add_argument('-l', type=str, default=None)
_p.add_argument('-p', action='store_true', default=False)

def run():
    "Launch ipyhermes REPL."
    args, rest = _p.parse_known_args()
    flags = (['-r', str(args.r)] if args.r is not None else []) + \
            (['-l', args.l] if args.l else []) + \
            (['-p'] if args.p else [])
    if flags: os.environ['IPYTHONNG_FLAGS'] = ' '.join(flags)
    start_ipython(argv=['--ext','ipythonng','--ext','safepyrun',
                        '--ext','pshnb','--ext','ipyhermes',
                        '--HistoryManager.db_log_output=True',
                        '--no-confirm-exit','--no-banner'] + rest)
