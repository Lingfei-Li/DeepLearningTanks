
import pygame
import scipy as sp



class CompleteState:

    def __init__(self, player, enemies, bullets, map, castle, bonus):
        # players, enemies, bullets, level, castle, bonus
        self.player = player
        self.enemies = enemies
        self.bullets = bullets
        self.map = map
        self.castle = castle
        self.bonus = bonus


def getRawState(screen):
    img = pygame.surfarray.array2d(screen)
    return sp.misc.imresize(img[:208, :208], (104, 104))
