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
import numpy as np


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
    def __init__(self):
        self.featureNum = 9


    @staticmethod
    def calcPosDiff(direction, step):
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

    @staticmethod
    def calcNextPos(position, direction, step):
        x = position.centerx
        y = position.centery
        dx, dy = SimpleExtractor.calcPosDiff(direction, step)
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
        player = state.player
        next_x, next_y = self.calcNextPos(player.rect, action - 1, 10.0)

        vertical, horizontal = 0, 0

        for bullet in state.bullets:
            dist = math.hypot(bullet.rect.centerx - next_x, bullet.rect.centery - next_y)
            if dist is not None:
                if abs(bullet.rect.centery - next_y) < 50:
                    if (player.direction == Const.DIR_UP or player.direction == Const.DO_DOWN) and player.direction == action-1:
                        horizontal += 0.5/dist
                    else:
                        horizontal += 1/dist
                elif abs(bullet.rect.centerx - next_x) < 50:
                    if (player.direction == Const.DIR_LEFT or player.direction == Const.DO_RIGHT) and player.direction == action-1:
                        vertical += 0.5/dist
                    else:
                        vertical += 1/dist

        return [horizontal, vertical]

    def probEnemy(self, state, action):
        player = state.player
        next_x, next_y = self.calcNextPos(player.rect, action - 1, 10.0)

        vertical, horizontal = 0, 0

        for enemy in state.enemies:
            dist = math.hypot(enemy.rect.centerx - next_x, enemy.rect.centery - next_y)
            if dist is not None:
                if abs(enemy.rect.centery - next_y) < 50:
                    if (player.direction == Const.DIR_UP or player.direction == Const.DO_DOWN) and player.direction == action-1:
                        horizontal += 0.5/dist
                    else:
                        horizontal += 1/dist
                elif abs(enemy.rect.centerx - next_x) < 50:
                    if (player.direction == Const.DIR_LEFT or player.direction == Const.DO_RIGHT) and player.direction == action-1:
                        vertical += 0.5/dist
                    else:
                        vertical += 1/dist

        return [horizontal, vertical]

    # def nearEdge(self, state, action):
    #     player = state.player
    #     next_x, next_y = self.calcNextPos(player.rect, action-1, 10.0)
    #
    #     horizon = min(abs(next_x), abs(next_x-480))
    #     vert = min(abs(next_y), abs(next_y-416))
    #     #return horizon/50 + vert/50
    #
    #     if abs(next_x) < 50 or abs(480-next_x) < 50 or abs(next_y) < 50 or abs(416-next_y) < 50:
    #         return [1]
    #     return [0]

    def nearEdge(self, state, action):
        player = state.player
        next_x, next_y = self.calcNextPos(player.rect, action-1, 10.0)

        res = [0, 0, 0, 0]
        if next_x < 20:
            res[0] = 1
        if 480-next_x < 20:
            res[1] = 1
        if next_y < 20:
            res[2] = 1
        if 416-next_y < 20:
            res[3] = 1
        return res

    def onRightSide(self, state, action):
        player = state.player
        next_x, next_y = self.calcNextPos(player.rect, action-1, 10.0)
        if next_x > 240:
            return [1]
        return [0]

    def moveFoward(self, state, action):
        player = state.player
        if player.direction == action-1:
            return [1]
        return [0]

    def hitEnemy(self, state, action):

        player = state.player
        next_x, next_y = player.rect.centerx, player.rect.centery

        prob = 0.0
        for enemy in state.enemies:
            dist = math.hypot(enemy.rect.centerx - next_x, enemy.rect.centery - next_y)
            if dist is not None:
                if abs(enemy.rect.centery - next_y) < 100:
                    if enemy.rect.centerx < next_x:     #enemy is to the left
                        if player.direction == Const.DIR_LEFT and action == Const.DO_FIRE:
                            return 1
                        elif  action == Const.DO_LEFT:
                            return 0.5
                    elif enemy.rect.centerx > next_x:
                        if player.direction == Const.DIR_RIGHT and action == Const.DO_FIRE:
                            return 1
                        elif action == Const.DO_RIGHT:
                            return 0.5
                elif abs(enemy.rect.centerx - next_x) < 100:
                    if enemy.rect.centery < next_y:
                        if player.direction == Const.DIR_UP and action == Const.DO_FIRE:
                            return 1
                        elif action == Const.DO_UP:
                            return 0.5
                    elif enemy.rect.centery > next_y:
                        if player.direction == Const.DIR_DOWN and action == Const.DO_FIRE:
                            return 1
                        elif action == Const.DO_DOWN:
                            return 0.5
        return 0

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        features = util.Counter()

        features["bias"] = 1.0
        features["horizontalEnemy"], features["verticalEnemy"] = self.probEnemy(state, action)
        features["horizontalBullet"], features["verticalBullet"]  = self.probBullet(state, action)
        features["edge"] = self.nearEdge(state, action)
        #features["hitEnemy"] = self.hitEnemy(state, action)
        # features["moveFoward"] = self.moveFoward(state, action)
        # features["onRightSide"] = self.onRightSide(state, action)
        #print(features["hitEnemy"])
        #features["vertSpace"] = self.verticalSpace(state, action)
        #features["horizonSpace"] = self.horizontalSpace(state, action)

        return [1.0] + self.probBullet(state, action) + self.probEnemy(state, action) + self.nearEdge(state, action)

        features.divideAll(10.0)
        return features


class PositionExtractor:
    @staticmethod
    def getFeatureNum():
        return 12

    @staticmethod
    def getFeatures(state):
        #3 enemies, x and y axis
        player_pos = np.zeros(2)
        player_pos[0], player_pos[1] = state.player.rect.centerx, state.player.rect.centery

        player_dist_to_edge = np.zeros(4)
        player_dist_to_edge[0], player_dist_to_edge[1] = state.player.rect.centerx, 480-state.player.rect.centerx
        player_dist_to_edge[2], player_dist_to_edge[3] = state.player.rect.centery, 416-state.player.rect.centery

        # enemy_pos = np.zeros(9)
        enemy_pos = np.zeros(3)
        for idx, enemy in enumerate(state.enemies):
            if idx > 2: break;
            enemy_pos[idx*3] = enemy.rect.centerx
            enemy_pos[idx*3+1] = enemy.rect.centery
            enemy_pos[idx*3+2] = enemy.direction

        # bullet_pos = np.zeros(12)
        bullet_pos = np.zeros(3)
        for idx, bullet in enumerate(state.bullets):
            if idx >= 1: break;
            bullet_pos[idx*3] = bullet.rect.centerx
            bullet_pos[idx*3+1] = bullet.rect.centery
            bullet_pos[idx*3+2] = bullet.direction

        return np.append(np.append(player_pos, player_dist_to_edge), np.append(enemy_pos, bullet_pos)).tolist()

class DangerExtractorInstance():

    @staticmethod
    def upsideClode(x1, y1, x2, y2):
        return abs(x1-x2) < 50 and y1 < y2

    @staticmethod
    def downsideClode(x1, y1, x2, y2):
        return abs(x1-x2) < 50 and y1 > y2

    @staticmethod
    def leftsideClode(x1, y1, x2, y2):
        return abs(y1-y2) < 50 and x1 < x2

    @staticmethod
    def rightsideClode(x1, y1, x2, y2):
        return abs(y1-y2) < 50 and x1 > x2

    @staticmethod
    def probBullet(state, action):
        player = state.player
        next_x = player.rect.centerx
        next_y = player.rect.centery

        next_x, next_y = SimpleExtractor.calcNextPos(player.rect, action - 1, 10.0)

        res = [0,0,0,0]

        for bullet in state.bullets:
            dist = math.hypot(bullet.rect.centerx - next_x, bullet.rect.centery - next_y)
            x1 = player.rect.centerx
            y1 = player.rect.centery
            x2 = bullet.rect.centerx
            y2 = bullet.rect.centery
            if dist is not None:
                if DangerExtractor.upsideClode(x1, y1, x2, y2):
                    res[0] = 1
                elif DangerExtractor.rightsideClode(x1, y1, x2, y2):
                    res[1] = 1
                elif DangerExtractor.downsideClode(x1, y1, x2, y2):
                    res[2] = 1
                elif DangerExtractor.leftsideClode(x1, y1, x2, y2):
                    res[3] = 1
        return res

    @staticmethod
    def probEnemy(state):
        player = state.player
        next_x = player.rect.centerx
        next_y = player.rect.centery

        vertical, horizontal = 0, 0

        for enemy in state.enemies:
            dist = math.hypot(enemy.rect.centerx - next_x, enemy.rect.centery - next_y)
            if dist is not None:
                if abs(enemy.rect.centery - next_y) < 50 and dist < 300:
                    horizontal += 1
                elif abs(enemy.rect.centerx - next_x) < 50 and dist < 300:
                    vertical += 1

        return [horizontal, vertical]

    @staticmethod
    def nearEdge(state, action):
        player = state.player
        next_x = player.rect.centerx
        next_y = player.rect.centery

        next_x, next_y = SimpleExtractor.calcNextPos(player.rect, action - 1, 10.0)

        # return [next_x, 480-next_x, next_y, 416-next_y]
        res = [0, 0, 0, 0]
        if next_x < 20:
            res[0] = 1
        if 480-next_x < 20:
            res[1] = 1
        if next_y < 20:
            res[2] = 1
        if 416-next_y < 20:
            res[3] = 1
        return res

    def getFeatureNum(self):
        return 8

    def getFeatures(self, state, action):
        return self.probBullet(state, action) + self.nearEdge(state, action)

class DangerExtractor:

    @staticmethod
    def upsideClode(x1, y1, x2, y2):
        return abs(x1-x2) < 50 and y1 < y2

    @staticmethod
    def downsideClode(x1, y1, x2, y2):
        return abs(x1-x2) < 50 and y1 > y2

    @staticmethod
    def leftsideClode(x1, y1, x2, y2):
        return abs(y1-y2) < 50 and x1 < x2

    @staticmethod
    def rightsideClode(x1, y1, x2, y2):
        return abs(y1-y2) < 50 and x1 > x2

    @staticmethod
    def probBullet(state):
        player = state.player
        next_x = player.rect.centerx
        next_y = player.rect.centery


        res = [0,0,0,0]

        for bullet in state.bullets:
            dist = math.hypot(bullet.rect.centerx - next_x, bullet.rect.centery - next_y)
            x1 = player.rect.centerx
            y1 = player.rect.centery
            x2 = bullet.rect.centerx
            y2 = bullet.rect.centery
            if dist is not None:
                if DangerExtractor.upsideClode(x1, y1, x2, y2):
                    res[0] = 1
                elif DangerExtractor.rightsideClode(x1, y1, x2, y2):
                    res[1] = 1
                elif DangerExtractor.downsideClode(x1, y1, x2, y2):
                    res[2] = 1
                elif DangerExtractor.leftsideClode(x1, y1, x2, y2):
                    res[3] = 1
        return res

    @staticmethod
    def probEnemy(state):
        player = state.player
        next_x = player.rect.centerx
        next_y = player.rect.centery

        vertical, horizontal = 0, 0

        for enemy in state.enemies:
            dist = math.hypot(enemy.rect.centerx - next_x, enemy.rect.centery - next_y)
            if dist is not None:
                if abs(enemy.rect.centery - next_y) < 50 and dist < 300:
                    horizontal += 1
                elif abs(enemy.rect.centerx - next_x) < 50 and dist < 300:
                    vertical += 1

        return [horizontal, vertical]

    @staticmethod
    def nearEdge(state):
        player = state.player
        next_x = player.rect.centerx
        next_y = player.rect.centery

        # return [next_x, 480-next_x, next_y, 416-next_y]
        res = [0, 0, 0, 0]
        if next_x < 20:
            res[0] = 1
        if 480-next_x < 20:
            res[1] = 1
        if next_y < 20:
            res[2] = 1
        if 416-next_y < 20:
            res[3] = 1
        return res

    @staticmethod
    def getFeatureNum():
        return 8

    @staticmethod
    def getFeatures(state):
        return DangerExtractor.probBullet(state) + DangerExtractor.nearEdge(state)
        return DangerExtractor.probEnemy(state) + DangerExtractor.probBullet(state) + DangerExtractor.nearEdge(state)

