"""
Main entry point for running from command line.
"""

import os
import sys
import multiprocessing as mp

import argparse

from logging import getLogger,disable

from helpfunc.logger import setup_logger
from config import Config

logger = getLogger(__name__)

CMD_LIST = ['self', 'opt', 'eval', 'sl', 'uci', 'all']

_PATH_ = os.path.dirname(os.path.dirname(__file__))


if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

def create_parser():
    """
    Parses each of the arguments from the command line
    :return ArgumentParser representing the command line arguments that were supplied to the command line:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", help="what to do", choices=CMD_LIST)
    parser.add_argument("--new", help="run from new best model", action="store_true")
    parser.add_argument("--type", help="use normal setting", default="normal")
    parser.add_argument("--total-step", help="set TrainerConfig.start_total_steps", type=int)
    return parser


def setup(config: Config, args):
    """
    Sets up a new config by creating the required directories and setting up logging.
    """
    config.opts.new = args.new
    if args.total_step is not None:
        config.trainer.start_total_steps = args.total_step
    config.resource.create_directories()
    setup_logger(config.resource.main_log_path)


def start():
    """
    Starts one of the processes based on command line arguments.

    :return : the worker class that was started
    """
    parser = create_parser()
    args = parser.parse_args()
    config_type = args.type

    if args.cmd == 'uci':
        disable(999999) # plz don't interfere with uci

    config = Config(config_type=config_type)
    setup(config, args)

    logger.info(f"config type: {config_type}")

    if args.cmd == 'self':
        from workers import self_play
        return self_play.start(config)
    elif args.cmd == 'opt':
        from workers import optimize
        return optimize.start(config)
    elif args.cmd == 'eval':
        from workers import evaluate
        return evaluate.start(config)
    elif args.cmd == 'sl':
        from workers import sl
        return sl.start(config)
    elif args.cmd == 'uci':
        from Game import uci
        return uci.start(config)
    elif args.cmd == 'all':
        from workers import evaluate
        from workers import play_and_optimize
        
        for i in range (0, 1):
            print("cycle" + str(i))
            play_and_optimize.start(config)
            i += 1
                    
        return evaluate.start(config)



if __name__ == "__main__":
    mp.set_start_method('spawn')
    sys.setrecursionlimit(10000)
    start()
