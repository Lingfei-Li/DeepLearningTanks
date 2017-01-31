# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Battle City Tanks game states"
import math

import util
from const import Const


class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()


class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether enemy will be hit
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def enemiesFaced(self, state):

        direction = state.player.direction
        facedEnemies = []
        for enemy in state.enemies:
            if direction == Const.DIR_UP:
                if enemy.rect.centery < state.player.rect.centery:
                    if enemy.rect.centerx < state.player.rect.centerx and enemy.direction == Const.DIR_RIGHT:
                        facedEnemies.append(enemy)
                    if enemy.rect.centerx > state.player.rect.centerx and enemy.direction == Const.DIR_LEFT:
                        facedEnemies.append(enemy)
            elif direction == Const.DIR_RIGHT:
                if enemy.rect.centerx > state.player.rect.centerx:
                    if enemy.rect.centery < state.player.rect.centery and enemy.direction == Const.DIR_DOWN:
                        facedEnemies.append(enemy)
                    if enemy.rect.centery > state.player.rect.centery and enemy.direction == Const.DIR_UP:
                        facedEnemies.append(enemy)
            elif direction == Const.DIR_DOWN:
                if enemy.rect.centery > state.player.rect.centery:
                    if enemy.rect.centerx < state.player.rect.centerx and enemy.direction == Const.DIR_RIGHT:
                        facedEnemies.append(enemy)
                    if enemy.rect.centerx > state.player.rect.centerx and enemy.direction == Const.DIR_LEFT:
                        facedEnemies.append(enemy)
            elif direction == Const.DIR_LEFT:
                if enemy.rect.centerx < state.player.rect.centerx:
                    if enemy.rect.centery < state.player.rect.centery and enemy.direction == Const.DIR_DOWN:
                        facedEnemies.append(enemy)
                    if enemy.rect.centery > state.player.rect.centery and enemy.direction == Const.DIR_UP:
                        facedEnemies.append(enemy)

        return facedEnemies

    def calcPosDiff(self, direction, step):
        dx = 0
        dy = 0
        if direction == Const.DIR_UP:
            dy = -step
        elif direction == Const.DIR_RIGHT:
            dx = step
        elif direction == Const.DIR_DOWN:
            dy = step
        elif direction == Const.DIR_LEFT:
            dx = -step
        return dx, dy

    def calcNextPos(self, position, direction, step):
        x = position.centerx
        y = position.centery
        dx, dy = self.calcPosDiff(direction, step)
        next_x, next_y = float(x + dx), float(y + dy)
        if next_x > 480:
            next_x = 480
        elif next_x < 0:
            next_x = 0
        if next_y > 416:
            next_y = 416
        elif next_y < 0:
            next_y = 0
        return next_x, next_y

    def probBullet(self, state, action):
        bullet_prob = 0.0
        player = state.player
        next_x, next_y = self.calcNextPos(player.rect, action-1, 10.0)

        factor = 100
        for bullet in state.bullets:
            dist = math.hypot(bullet.rect.centerx - next_x, bullet.rect.centery - next_y)
            if dist is not None:
                if abs(bullet.rect.centery - next_y) < 50:
                    if bullet.rect.centerx < next_x and bullet.direction == Const.DIR_RIGHT:
                        if (player.direction == Const.DIR_UP or player.direction == Const.DO_DOWN) and player.direction == action-1:
                            bullet_prob += 0.1*factor/dist
                        else:
                            bullet_prob += factor/dist
                    elif bullet.rect.centerx > next_x and bullet.direction == Const.DIR_LEFT:
                        if (player.direction == Const.DIR_UP or player.direction == Const.DO_DOWN) and player.direction == action-1:
                            bullet_prob += 0.1*factor/dist
                        else:
                            bullet_prob += factor/dist
                elif abs(bullet.rect.centerx-next_x) < 50:
                    if bullet.rect.centery < next_y and bullet.direction == Const.DIR_DOWN:
                        if (player.direction == Const.DIR_LEFT or player.direction == Const.DO_RIGHT) and player.direction == action-1:
                            bullet_prob += 0.1*factor/dist
                        else:
                            bullet_prob += factor/dist
                    elif bullet.rect.centery > next_y and bullet.direction == Const.DIR_UP:
                        if (player.direction == Const.DIR_LEFT or player.direction == Const.DO_RIGHT) and player.direction == action-1:
                            bullet_prob += 0.1*factor/dist
                        else:
                            bullet_prob += factor/dist

        return bullet_prob/factor

    def probEnemy(self, state, action):
        player = state.player
        next_x, next_y = self.calcNextPos(player.rect, action-1, 10.0)

        enemy_prob = 0.0
        factor = 100

        for enemy in state.enemies:
            dist = math.hypot(enemy.rect.centerx - next_x, enemy.rect.centery - next_y)
            if dist is not None:
                if abs(enemy.rect.centery - next_y) < 50:
                    if enemy.rect.centerx < next_x and enemy.direction == Const.DIR_RIGHT:
                        if (player.direction == Const.DIR_UP or player.direction == Const.DO_DOWN) and player.direction == action-1:
                            enemy_prob += 0.1*factor/dist
                        else:
                            enemy_prob += factor/dist
                    elif enemy.rect.centerx > next_x and enemy.direction == Const.DIR_LEFT:
                        if (player.direction == Const.DIR_UP or player.direction == Const.DO_DOWN) and player.direction == action-1:
                            enemy_prob += 0.1*factor/dist
                        else:
                            enemy_prob += factor/dist
                    else:
                        enemy_prob += 0.5*factor/dist
                elif abs(enemy.rect.centerx - next_x) < 50:
                    if enemy.rect.centery < next_y and enemy.direction == Const.DIR_DOWN:
                        if (player.direction == Const.DIR_LEFT or player.direction == Const.DO_RIGHT) and player.direction == action-1:
                            enemy_prob += 0.1*factor/dist
                        else:
                            enemy_prob += factor/dist
                    elif enemy.rect.centery > next_y and enemy.direction == Const.DIR_UP:
                        if (player.direction == Const.DIR_LEFT or player.direction == Const.DO_RIGHT) and player.direction == action-1:
                            enemy_prob += 0.1*factor/dist
                        else:
                            enemy_prob += factor/dist
                    else:
                        enemy_prob += 0.5*factor/dist

        return enemy_prob/factor

    def probSpace(self, state, action):
        player = state.player
        x = player.rect.centerx
        y = player.rect.centery
        dx, dy = self.calcPosDiff(action-1, 10.0)
        next_x, next_y = float(x + dx), float(y + dy)

        return abs(abs(next_x)-abs(480-next_x)) + abs(abs(next_y)-abs(416-next_y))




    def probHitEnemy(self, state, action):
        facedEnemies = self.enemiesFaced(state)
        prob = 0.0
        for enemy in facedEnemies:
            dist = util.dist(state.player, enemy)
            if dist is not None:
                prob += 120/dist
                # prob += 480/abs(enemy.rect.centerx - state.player.rect.centerx)
                # prob += 416/abs(enemy.rect.centery - state.player.rect.centery)
        return prob


    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        features = util.Counter()

        features["bias"] = 1.0
        # features["hit"] = self.probHitEnemy(state, action)
        features["enemy"] = self.probEnemy(state, action)
        features["bullet"] = self.probBullet(state, action)
        # features["space"] = self.probSpace(state, action)

        features.divideAll(2000.0)
        return features
