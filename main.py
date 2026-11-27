import os
import argparse
from game.game import Game, read_game
from hsvi.generator import write_hsvi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game_cfg", help="input game configuration file, in yaml")
    # parser.add_argument("algo", help="algorithm to use", choices=["hsvi", "sab"])

    algo_parsers = parser.add_subparsers(
        title="algorithms", dest="algo", description="valid algorithms", help="additional help"
    )
    parser_hsvi = algo_parsers.add_parser(
        "hsvi", help="Generate a file for the HSVI solver"
    )

    parser_sab = algo_parsers.add_parser("sab", help="Solve the game using SAB")
    parser_sab.add_argument("resolution", type=int, help="discretization resolution")

    args = parser.parse_args()

    path = args.game_cfg
    game = read_game(path)

    if args.algo == "hsvi":
        filename = os.path.basename(path).split(".")[0]
        print(f"Generating HSVI input file from spec into hsvi/{filename}.posg ...")
        write_hsvi(filename, game)

    if args.algo == "sab":
        print("Solving game from spec with SAB ...")
