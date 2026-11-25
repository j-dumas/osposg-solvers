import os
import argparse
from game.game import Game
from game.game_generator import read_game, write_hsvi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game_cfg", help="input game configuration file, in yaml")
    parser.add_argument("-o", help="output hsvi game format", action="store_true")
    args = parser.parse_args()

    path = args.game_cfg
    game = read_game(path)

    if args.o:
        write_hsvi(os.path.basename(path).split(".")[0], game)
