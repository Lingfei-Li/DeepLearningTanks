
import random

def argMax(values):
    maxIndex = values.index(max(values))
    return maxIndex, values


def flipCoin(p):
    r = random.random()
    return r < p

