from game_gridworld import *


def gym_make(game_name):
    if game_name == "game_gridworld":
        return GridWorldEnv()
    return gym.make(game_name)


