#!/usr/bin/python
# coding=utf-8

import os, pygame, time, random, uuid, sys, util, agents, state, pgAgents
from optparse import OptionParser
from const import Const
from featureExtractors import  PositionExtractor
from featureExtractors import  DangerExtractor
import scipy.misc as smp
import numpy as np
import scipy as sp


class myRect(pygame.Rect):
    """ Add type property """
    def __init__(self, left, top, width, height, type):
        pygame.Rect.__init__(self, left, top, width, height)
        self.type = type

class Timer(object):
    def __init__(self):
        self.timers = []

    def add(self, interval, f, repeat = -1):
        options = {
            "interval"	: interval,
            "callback"	: f,
            "repeat"		: repeat,
            "times"			: 0,
            "time"			: 0,
            "uuid"			: uuid.uuid4()
        }
        self.timers.append(options)

        return options["uuid"]

    def destroy(self, uuid_nr):
        for timer in self.timers:
            if timer["uuid"] == uuid_nr:
                self.timers.remove(timer)
                return

    def update(self, time_passed):
        for timer in self.timers:
            timer["time"] += time_passed
            if timer["time"] > timer["interval"]:
                timer["time"] -= timer["interval"]
                timer["times"] += 1
                if timer["repeat"] > -1 and timer["times"] == timer["repeat"]:
                    self.timers.remove(timer)
                try:
                    timer["callback"]()
                except:
                    try:
                        self.timers.remove(timer)
                    except:
                        pass

class Castle:
    """ Player's castle/fortress """

    (STATE_STANDING, STATE_DESTROYED, STATE_EXPLODING) = range(3)

    def __init__(self):

        global sprites

        # images
        self.img_undamaged = sprites.subsurface(0, 15, 16, 16)
        self.img_destroyed = sprites.subsurface(16, 15, 16, 16)

        # init position
        # self.rect = pygame.Rect(12*16, 24*16, 32, 32)
        self.rect = pygame.Rect(-32, -32, 32, 32)

        # start w/ undamaged and shiny castle
        self.rebuild()

    def draw(self):
        """ Draw castle """
        global screen

        screen.blit(self.image, self.rect.topleft)

        if self.state == self.STATE_EXPLODING:
            if not self.explosion.active:
                self.state = self.STATE_DESTROYED
                del self.explosion
            else:
                self.explosion.draw()

    def rebuild(self):
        """ Reset castle """
        self.state = self.STATE_STANDING
        self.image = self.img_undamaged
        self.active = True

    def destroy(self):
        """ Destroy castle """
        self.state = self.STATE_EXPLODING
        self.explosion = Explosion(self.rect.topleft)
        self.image = self.img_destroyed
        self.active = False

class Bonus:
    """ Various power-ups
    When bonus is spawned, it begins flashing and after some time dissapears

    Available bonusses:
        grenade	: Picking up the grenade power up instantly wipes out ever enemy presently on the screen, including Armor Tanks regardless of how many times you've hit them. You do not, however, get credit for destroying them during the end-stage bonus points.
        helmet	: The helmet power up grants you a temporary force field that makes you invulnerable to enemy shots, just like the one you begin every stage with.
        shovel	: The shovel power up turns the walls around your fortress from brick to stone. This makes it impossible for the enemy to penetrate the wall and destroy your fortress, ending the game prematurely. The effect, however, is only temporary, and will wear off eventually.
        star		: The star power up grants your tank with new offensive power each time you pick one up, up to three times. The first star allows you to fire your bullets as fast as the power tanks can. The second star allows you to fire up to two bullets on the screen at one time. And the third star allows your bullets to destroy the otherwise unbreakable steel walls. You carry this power with you to each new stage until you lose a life.
        tank		: The tank power up grants you one extra life. The only other way to get an extra life is to score 20000 points.
        timer		: The timer power up temporarily freezes time, allowing you to harmlessly approach every tank and destroy them until the time freeze wears off.
    """

    # bonus types
    (BONUS_GRENADE, BONUS_HELMET, BONUS_SHOVEL, BONUS_STAR, BONUS_TANK, BONUS_TIMER) = range(6)

    def __init__(self, level):

        global sprites

        # to know where to place
        self.level = level

        # bonus lives only for a limited period of time
        self.active = True

        # blinking state
        self.visible = True


        self.rect = pygame.Rect(random.randint(0, 416-32), random.randint(0, 416-32), 32, 32)

        self.bonus = random.choice([
            self.BONUS_GRENADE,
            self.BONUS_HELMET,
            self.BONUS_SHOVEL,
            self.BONUS_STAR,
            self.BONUS_TANK,
            self.BONUS_TIMER
        ])

        self.image = sprites.subsurface(16*self.bonus, 32, 16, 15)

    def draw(self):
        """ draw bonus """
        global screen
        if self.visible:
            screen.blit(self.image, self.rect.topleft)

    def toggleVisibility(self):
        """ Toggle bonus visibility """
        self.visible = not self.visible


class Bullet:
    # direction constants
    (DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT) = range(4)

    # bullet's stated
    (STATE_REMOVED, STATE_ACTIVE, STATE_EXPLODING) = range(3)

    (OWNER_PLAYER, OWNER_ENEMY) = range(2)

    def __init__(self, level, position, direction, damage = 100, speed = 5):

        global sprites

        self.level = level
        self.direction = direction
        self.damage = damage
        self.owner = None
        self.owner_class = None

        # 1-regular everyday normal bullet
        # 2-can destroy steel
        self.power = 1

        self.image = sprites.subsurface(75, 74, 3, 4)

        # position is player's top left corner, so we'll need to
        # recalculate a bit. also rotate image itself.
        if direction == self.DIR_UP:
            self.rect = pygame.Rect(position[0] + 10, position[1] - 4, 3, 4)
        elif direction == self.DIR_RIGHT:
            self.image = pygame.transform.rotate(self.image, 270)
            self.rect = pygame.Rect(position[0] + 13, position[1] + 5, 4, 3)
        elif direction == self.DIR_DOWN:
            self.image = pygame.transform.rotate(self.image, 180)
            self.rect = pygame.Rect(position[0] + 5, position[1] + 13, 3, 4)
        elif direction == self.DIR_LEFT:
            self.image = pygame.transform.rotate(self.image, 90)
            self.rect = pygame.Rect(position[0] - 4 , position[1] + 5, 4, 3)

        self.explosion_images = [
            sprites.subsurface(0, 40, 16, 16),
            sprites.subsurface(16, 40, 16, 16),
        ]

        self.speed = speed

        self.state = self.STATE_ACTIVE

    def draw(self):
        """ draw bullet """
        global screen
        if self.state == self.STATE_ACTIVE:
            screen.blit(self.image, self.rect.topleft)
        # elif self.state == self.STATE_EXPLODING:
        #     self.explosion.draw()

    def update(self):
        global castle, players, enemies, bullets

        if self.state == self.STATE_EXPLODING:
            if not self.explosion.active:
                self.destroy()
                del self.explosion

        if self.state != self.STATE_ACTIVE:
            return

        """ move bullet """
        if self.direction == self.DIR_UP:
            self.rect.topleft = [self.rect.left, self.rect.top - self.speed]
            if self.rect.top < 0:
                if play_sounds and self.owner == self.OWNER_PLAYER:
                    sounds["steel"].play()
                self.explode()
                return
        elif self.direction == self.DIR_RIGHT:
            self.rect.topleft = [self.rect.left + self.speed, self.rect.top]
            if self.rect.left > (208 - self.rect.width):
                if play_sounds and self.owner == self.OWNER_PLAYER:
                    sounds["steel"].play()
                self.explode()
                return
        elif self.direction == self.DIR_DOWN:
            self.rect.topleft = [self.rect.left, self.rect.top + self.speed]
            if self.rect.top > (208 - self.rect.height):
                if play_sounds and self.owner == self.OWNER_PLAYER:
                    sounds["steel"].play()
                self.explode()
                return
        elif self.direction == self.DIR_LEFT:
            self.rect.topleft = [self.rect.left - self.speed, self.rect.top]
            if self.rect.left < 0:
                if play_sounds and self.owner == self.OWNER_PLAYER:
                    sounds["steel"].play()
                self.explode()
                return

        has_collided = False

        # check for collisions with walls. one bullet can destroy several (1 or 2)
        # tiles but explosion remains 1
        rects = self.level.obstacle_rects
        collisions = self.rect.collidelistall(rects)
        if collisions != []:
            for i in collisions:
                if self.level.hitTile(rects[i].topleft, self.power, self.owner == self.OWNER_PLAYER):
                    has_collided = True
        if has_collided:
            self.explode()
            return

        # check for collisions with other bullets
        for bullet in bullets:
            if self.state == self.STATE_ACTIVE and bullet.owner != self.owner and bullet != self and self.rect.colliderect(bullet.rect):
                self.destroy()
                self.explode()
                return

        # check for collisions with players
        for player in players:
            if player.state == player.STATE_ALIVE and self.rect.colliderect(player.rect):
                if player.bulletImpact(self.owner == self.OWNER_PLAYER, self.damage, self.owner_class):
                    self.destroy()
                    return

        # check for collisions with enemies
        for enemy in enemies:
            if enemy.state == enemy.STATE_ALIVE and self.rect.colliderect(enemy.rect):
                if enemy.bulletImpact(self.owner == self.OWNER_ENEMY, self.damage, self.owner_class):
                    self.destroy()
                    return

        # check for collision with castle
        if castle.active and self.rect.colliderect(castle.rect):
            castle.destroy()
            self.destroy()
            return

    def explode(self):
        """ start bullets's explosion """
        global screen
        if self.state != self.STATE_REMOVED:
            self.state = self.STATE_EXPLODING
            self.explosion = Explosion([self.rect.left-13, self.rect.top-13], None, self.explosion_images)

    def destroy(self):
        self.state = self.STATE_REMOVED


class Label:
    def __init__(self, position, text = "", duration = None):

        self.position = position

        self.active = True

        self.text = text

        self.font = pygame.font.SysFont("Arial", 13)

        if duration != None:
            gtimer.add(duration/options["game_speed"], lambda :self.destroy(), 1)

    def draw(self):
        """ draw label """
        global screen
        screen.blit(self.font.render(self.text, False, (200,200,200)), [self.position[0]+4, self.position[1]+8])

    def destroy(self):
        self.active = False


class Explosion:
    def __init__(self, position, interval = None, images = None):

        global sprites

        self.position = [position[0]-16, position[1]-16]
        self.active = True

        if interval == None:
            interval = 100

        if images == None:
            images = [
                sprites.subsurface(0, 80, 32, 32),
                sprites.subsurface(32, 80, 32, 32),
                sprites.subsurface(64, 80, 32, 32)
            ]

        images.reverse()

        self.images = [] + images

        self.image = self.images.pop()

        gtimer.add(interval/options["game_speed"], lambda :self.update(), len(self.images) + 1)

    def draw(self):
        global screen
        """ draw current explosion frame """
        screen.blit(self.image, self.position)

    def update(self):
        """ Advace to the next image """
        if len(self.images) > 0:
            self.image = self.images.pop()
        else:
            self.active = False

class Level:

    # tile constants
    (TILE_EMPTY, TILE_BRICK, TILE_STEEL, TILE_WATER, TILE_GRASS, TILE_FROZE) = range(6)

    # tile width/height in px
    TILE_SIZE = 16

    def __init__(self, level_nr = None):
        """ There are total 35 different levels. If level_nr is larger than 35, loop over
        to next according level so, for example, if level_nr ir 37, then load level 2 """

        global sprites

        # max number of enemies simultaneously  being on map
        self.max_active_enemies = 3

        tile_images = [
            pygame.Surface((8, 8)),
            sprites.subsurface(48, 64, 8, 8),
            sprites.subsurface(48, 72, 8, 8),
            sprites.subsurface(56, 72, 8, 8),
            sprites.subsurface(64, 64, 8, 8),
            sprites.subsurface(64, 64, 8, 8),
            sprites.subsurface(72, 64, 8, 8),
            sprites.subsurface(64, 72, 8, 8)
        ]
        self.tile_empty = tile_images[0]
        self.tile_brick = tile_images[1]
        self.tile_steel = tile_images[2]
        self.tile_grass = tile_images[3]
        self.tile_water = tile_images[4]
        self.tile_water1= tile_images[4]
        self.tile_water2= tile_images[5]
        self.tile_froze = tile_images[6]

        self.obstacle_rects = []

        level_nr = 1 if level_nr == None else level_nr%35
        if level_nr == 0:
            level_nr = 35

        self.loadLevel(level_nr)

        # tiles' rects on map, tanks cannot move over
        self.obstacle_rects = []

        # update these tiles
        self.updateObstacleRects()

        gtimer.add(400/options["game_speed"], lambda :self.toggleWaves())

    def hitTile(self, pos, power = 1, sound = False):
        """
            Hit the tile
            @param pos Tile's x, y in px
            @return True if bullet was stopped, False otherwise
        """

        global play_sounds, sounds

        for tile in self.mapr:
            if tile.topleft == pos:
                if tile.type == self.TILE_BRICK:
                    if play_sounds and sound:
                        sounds["brick"].play()
                    self.mapr.remove(tile)
                    self.updateObstacleRects()
                    return True
                elif tile.type == self.TILE_STEEL:
                    if play_sounds and sound:
                        sounds["steel"].play()
                    if power == 2:
                        self.mapr.remove(tile)
                        self.updateObstacleRects()
                    return True
                else:
                    return False

    def toggleWaves(self):
        """ Toggle water image """
        if self.tile_water == self.tile_water1:
            self.tile_water = self.tile_water2
        else:
            self.tile_water = self.tile_water1


    def loadLevel(self, level_nr = 1):
        """ Load specified level
        @return boolean Whether level was loaded
        """
        filename = "levels/"+options["level_type"]+"/"+str(level_nr)
        if not os.path.isfile(filename):
            return False
        level = []
        f = open(filename, "r")
        data = f.read().split("\n")
        self.mapr = []
        x, y = 0, 0
        for row in data:
            for ch in row:
                if ch == "#":
                    self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_BRICK))
                elif ch == "@":
                    self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_STEEL))
                elif ch == "~":
                    self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_WATER))
                elif ch == "%":
                    self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_GRASS))
                elif ch == "-":
                    self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_FROZE))
                x += self.TILE_SIZE
            x = 0
            y += self.TILE_SIZE
        return True


    def draw(self, tiles = None):
        """ Draw specified map on top of existing surface """

        global screen

        if tiles == None:
            tiles = [TILE_BRICK, TILE_STEEL, TILE_WATER, TILE_GRASS, TILE_FROZE]

        for tile in self.mapr:
            if tile.type in tiles:
                if tile.type == self.TILE_BRICK:
                    screen.blit(self.tile_brick, tile.topleft)
                elif tile.type == self.TILE_STEEL:
                    screen.blit(self.tile_steel, tile.topleft)
                elif tile.type == self.TILE_WATER:
                    screen.blit(self.tile_water, tile.topleft)
                elif tile.type == self.TILE_FROZE:
                    screen.blit(self.tile_froze, tile.topleft)
                elif tile.type == self.TILE_GRASS:
                    screen.blit(self.tile_grass, tile.topleft)

    def updateObstacleRects(self):
        """ Set self.obstacle_rects to all tiles' rects that players can destroy
        with bullets """

        global castle

        self.obstacle_rects = [castle.rect]

        for tile in self.mapr:
            if tile.type in (self.TILE_BRICK, self.TILE_STEEL, self.TILE_WATER):
                self.obstacle_rects.append(tile)

    def buildFortress(self, tile):
        """ Build walls around castle made from tile """

        positions = [
            (11*self.TILE_SIZE, 23*self.TILE_SIZE),
            (11*self.TILE_SIZE, 24*self.TILE_SIZE),
            (11*self.TILE_SIZE, 25*self.TILE_SIZE),
            (14*self.TILE_SIZE, 23*self.TILE_SIZE),
            (14*self.TILE_SIZE, 24*self.TILE_SIZE),
            (14*self.TILE_SIZE, 25*self.TILE_SIZE),
            (12*self.TILE_SIZE, 23*self.TILE_SIZE),
            (13*self.TILE_SIZE, 23*self.TILE_SIZE)
        ]

        obsolete = []

        for i, rect in enumerate(self.mapr):
            if rect.topleft in positions:
                obsolete.append(rect)
        for rect in obsolete:
            self.mapr.remove(rect)

        for pos in positions:
            self.mapr.append(myRect(pos[0], pos[1], self.TILE_SIZE, self.TILE_SIZE, tile))

        self.updateObstacleRects()

class Tank:

    # possible directions
    (DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT) = range(4)

    # states
    (STATE_SPAWNING, STATE_DEAD, STATE_ALIVE, STATE_EXPLODING) = range(4)

    # sides
    (SIDE_PLAYER, SIDE_ENEMY) = range(2)

    def __init__(self, level, side, position = None, direction = None, filename = None):

        global sprites

        # health. 0 health means dead
        self.health = 100

        # tank can't move but can rotate and shoot
        self.paralised = False

        # tank can't do anything
        self.paused = False

        # tank is protected from bullets
        self.shielded = False

        # px per move
        self.speed = 2

        # how many bullets can tank fire simultaneously
        self.max_active_bullets = 1

        # friend or foe
        self.side = side

        # flashing state. 0-off, 1-on
        self.flash = 0

        # 0 - no superpowers
        # 1 - faster bullets
        # 2 - can fire 2 bullets
        # 3 - can destroy steel
        self.superpowers = 0

        # each tank can pick up 1 bonus
        self.bonus = None

        # navigation keys: fire, up, right, down, left
        self.controls = [pygame.K_SPACE, pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT]

        # currently pressed buttons (navigation only)
        self.pressed = [False] * 4

        self.shield_images = [
            sprites.subsurface(0, 48, 16, 16),
            sprites.subsurface(16, 48, 16, 16)
        ]
        self.shield_image = self.shield_images[0]
        self.shield_index = 0

        self.spawn_images = [
            sprites.subsurface(32, 48, 16, 16),
            sprites.subsurface(48, 48, 16, 16)
        ]
        self.spawn_image = self.spawn_images[0]
        self.spawn_index = 0

        self.level = level

        if position != None:
            self.rect = pygame.Rect(position, (26, 26))
        else:
            self.rect = pygame.Rect(0, 0, 26, 26)

        if direction == None:
            self.direction = random.choice([self.DIR_RIGHT, self.DIR_DOWN, self.DIR_LEFT])
        else:
            self.direction = direction

        self.state = self.STATE_SPAWNING

        # spawning animation
        self.timer_uuid_spawn = gtimer.add(100/options["game_speed"], lambda :self.toggleSpawnImage())

        # duration of spawning
        self.timer_uuid_spawn_end = gtimer.add(1000/options["game_speed"], lambda :self.endSpawning())

    def endSpawning(self):
        """ End spawning
        Player becomes operational
        """
        self.state = self.STATE_ALIVE
        gtimer.destroy(self.timer_uuid_spawn_end)


    def toggleSpawnImage(self):
        """ advance to the next spawn image """
        if self.state != self.STATE_SPAWNING:
            gtimer.destroy(self.timer_uuid_spawn)
            return
        self.spawn_index += 1
        if self.spawn_index >= len(self.spawn_images):
            self.spawn_index = 0
        self.spawn_image = self.spawn_images[self.spawn_index]

    def toggleShieldImage(self):
        """ advance to the next shield image """
        if self.state != self.STATE_ALIVE:
            gtimer.destroy(self.timer_uuid_shield)
            return
        if self.shielded:
            self.shield_index += 1
            if self.shield_index >= len(self.shield_images):
                self.shield_index = 0
            self.shield_image = self.shield_images[self.shield_index]


    def draw(self):
        """ draw tank """
        global screen
        if self.state == self.STATE_ALIVE:
            screen.blit(self.image, self.rect.topleft)
            if self.shielded:
                screen.blit(self.shield_image, [self.rect.left-3, self.rect.top-3])
        elif self.state == self.STATE_EXPLODING:
            self.explosion.draw()
        elif self.state == self.STATE_SPAWNING:
            screen.blit(self.spawn_image, self.rect.topleft)

    def explode(self):
        """ start tanks's explosion """
        if self.state != self.STATE_DEAD:
            self.state = self.STATE_EXPLODING
            self.explosion = Explosion(self.rect.topleft)

            if self.bonus:
                self.spawnBonus()

    def fire(self, forced = False):
        """ Shoot a bullet
        @param boolean forced. If false, check whether tank has exceeded his bullet quota. Default: True
        @return boolean True if bullet was fired, false otherwise
        """

        global bullets, labels

        if self.state != self.STATE_ALIVE:
            gtimer.destroy(self.timer_uuid_fire)
            return False

        if self.paused:
            return False

        if not forced:
            active_bullets = 0
            for bullet in bullets:
                if bullet.owner_class == self and bullet.state == bullet.STATE_ACTIVE:
                    active_bullets += 1
            if active_bullets >= self.max_active_bullets:
                return False

        bullet = Bullet(self.level, self.rect.topleft, self.direction)

        # if superpower level is at least 1
        if self.superpowers > 0:
            bullet.speed = 8

        # if superpower level is at least 3
        if self.superpowers > 2:
            bullet.power = 2

        if self.side == self.SIDE_PLAYER:
            bullet.owner = self.SIDE_PLAYER
        else:
            bullet.owner = self.SIDE_ENEMY
            self.bullet_queued = False

        bullet.owner_class = self
        bullets.append(bullet)
        return True

    def rotate(self, direction, fix_position = True):
        """ Rotate tank
        rotate, update image and correct position
        """
        self.direction = direction

        if direction == self.DIR_UP:
            self.image = self.image_up
        elif direction == self.DIR_RIGHT:
            self.image = self.image_right
        elif direction == self.DIR_DOWN:
            self.image = self.image_down
        elif direction == self.DIR_LEFT:
            self.image = self.image_left

        if fix_position:
            new_x = self.nearest(self.rect.left, 8) + 3
            new_y = self.nearest(self.rect.top, 8) + 3

            if abs(self.rect.left - new_x) < 5:
                self.rect.left = new_x

            if abs(self.rect.top - new_y) < 5:
                self.rect.top = new_y

    def turnAround(self):
        """ Turn tank into opposite direction """
        if self.direction in (self.DIR_UP, self.DIR_RIGHT):
            self.rotate(self.direction + 2, False)
        else:
            self.rotate(self.direction - 2, False)

    def update(self, time_passed):
        """ Update timer and explosion (if any) """
        if self.state == self.STATE_EXPLODING:
            if not self.explosion.active:
                self.state = self.STATE_DEAD
                del self.explosion

    def nearest(self, num, base):
        """ Round number to nearest divisible """
        return int(round(num / (base * 1.0)) * base)


    def bulletImpact(self, friendly_fire = False, damage = 100, tank = None):
        """ Bullet impact
        Return True if bullet should be destroyed on impact. Only enemy friendly-fire
        doesn't trigger bullet explosion
        """

        global play_sounds, sounds

        if self.shielded:
            return True

        if not friendly_fire:
            self.health -= damage
            if self.health < 1:
                if self.side == self.SIDE_ENEMY:
                    tank.trophies["enemy"+str(self.type)] += 1
                    points = (self.type+1) * 100
                    tank.score += points
                    if play_sounds:
                        sounds["explosion"].play()

                    labels.append(Label(self.rect.topleft, str(points), 500))
                # elif self.side == self.SIDE_PLAYER:
                #     self.score -= 1000
                    #self.score -= 0


                self.explode()
            return True

        if self.side == self.SIDE_ENEMY:
            return False
        elif self.side == self.SIDE_PLAYER:
            if not self.paralised:
                self.setParalised(True)
                self.timer_uuid_paralise = gtimer.add(10000/options["game_speed"], lambda :self.setParalised(False), 1)
            return True

    def setParalised(self, paralised = True):
        """ set tank paralise state
        @param boolean paralised
        @return None
        """
        if self.state != self.STATE_ALIVE:
            gtimer.destroy(self.timer_uuid_paralise)
            return
        self.paralised = paralised

class Enemy(Tank):

    (TYPE_BASIC, TYPE_FAST, TYPE_POWER, TYPE_ARMOR) = range(4)

    def __init__(self, level, type, position = None, direction = None, filename = None):

        Tank.__init__(self, level, type, position = None, direction = None, filename = None)

        global enemies, sprites

        # if true, do not fire
        self.bullet_queued = False

        # chose type on random
        if len(level.enemies_left) > 0:
            self.type = level.enemies_left.pop()
        else:
            self.state = self.STATE_DEAD
            return

        if self.type == self.TYPE_BASIC:
            self.speed = 1
        elif self.type == self.TYPE_FAST:
            self.speed = 3
        elif self.type == self.TYPE_POWER:
            self.superpowers = 1
        elif self.type == self.TYPE_ARMOR:
            self.health = 400

        # 1 in 5 chance this will be bonus carrier, but only if no other tank is
        if random.randint(1, 5) == 1:
            self.bonus = True
            for enemy in enemies:
                if enemy.bonus:
                    self.bonus = False
                    break

        images = [
            sprites.subsurface(32, 0, 13, 15),
            sprites.subsurface(48, 0, 13, 15),
            sprites.subsurface(64, 0, 13, 15),
            sprites.subsurface(80, 0, 13, 15),
            sprites.subsurface(32, 16, 13, 15),
            sprites.subsurface(48, 16, 13, 15),
            sprites.subsurface(64, 16, 13, 15),
            sprites.subsurface(80, 16, 13, 15)
        ]

        self.image = images[self.type+0]

        self.image_up = self.image;
        self.image_left = pygame.transform.rotate(self.image, 90)
        self.image_down = pygame.transform.rotate(self.image, 180)
        self.image_right = pygame.transform.rotate(self.image, 270)

        if self.bonus:
            self.image1_up = self.image_up;
            self.image1_left = self.image_left
            self.image1_down = self.image_down
            self.image1_right = self.image_right

            self.image2 = images[self.type+4]
            self.image2_up = self.image2;
            self.image2_left = pygame.transform.rotate(self.image2, 90)
            self.image2_down = pygame.transform.rotate(self.image2, 180)
            self.image2_right = pygame.transform.rotate(self.image2, 270)

        self.rotate(self.direction, False)

        if position == None:
            # self.rect.topleft = self.getFreeSpawningPosition()
            self.rect.topleft = self.getRandomSpawningPosition()
            if not self.rect.topleft:
                self.state = self.STATE_DEAD
                return

        # list of map coords where tank should go next
        self.path = self.generatePath(self.direction)

        # 1000 is duration between shots
        self.timer_uuid_fire = gtimer.add(1000/options["game_speed"], lambda :self.fire())

        # turn on flashing
        if self.bonus:
            self.timer_uuid_flash = gtimer.add(200/options["game_speed"], lambda :self.toggleFlash())

    def toggleFlash(self):
        """ Toggle flash state """
        if self.state not in (self.STATE_ALIVE, self.STATE_SPAWNING):
            gtimer.destroy(self.timer_uuid_flash)
            return
        self.flash = not self.flash
        if self.flash:
            self.image_up = self.image2_up
            self.image_right = self.image2_right
            self.image_down = self.image2_down
            self.image_left = self.image2_left
        else:
            self.image_up = self.image1_up
            self.image_right = self.image1_right
            self.image_down = self.image1_down
            self.image_left = self.image1_left
        self.rotate(self.direction, False)

    def spawnBonus(self):
        """ Create new bonus if needed """

        global bonuses

        if len(bonuses) > 0:
            return
        bonus = Bonus(self.level)
        bonuses.append(bonus)
        gtimer.add(500/options["game_speed"], lambda :bonus.toggleVisibility())
        gtimer.add(10000/options["game_speed"], lambda :bonuses.remove(bonus), 1)

    def getRandomSpawningPosition(self):

        global players, enemies

        available_positions = [
            [(self.level.TILE_SIZE * 2 - self.rect.width) / 2, (self.level.TILE_SIZE * 2 - self.rect.height) / 2],
            [6 * self.level.TILE_SIZE + (self.level.TILE_SIZE * 2 - self.rect.width) / 2, (self.level.TILE_SIZE * 2 - self.rect.height) / 2],
            [12 * self.level.TILE_SIZE + (self.level.TILE_SIZE * 2 - self.rect.width) / 2,  (self.level.TILE_SIZE * 2 - self.rect.height) / 2]
        ]
        while True:
            pos = [
                random.uniform(self.level.TILE_SIZE, 12*self.level.TILE_SIZE),
                random.uniform(self.level.TILE_SIZE, 12*self.level.TILE_SIZE)
            ]

            enemy_rect = pygame.Rect(pos, [26, 26])

            # collisions with other enemies
            collision = False
            for enemy in enemies:
                if enemy_rect.colliderect(enemy.rect):
                    collision = True
                    continue

            if collision:
                continue

            # collisions with players
            collision = False
            for player in players:
                if enemy_rect.colliderect(player.rect):
                    collision = True
                    continue

            if collision:
                continue

            return pos

    def getFreeSpawningPosition(self):

        global players, enemies

        available_positions = [
            [(self.level.TILE_SIZE * 2 - self.rect.width) / 2, (self.level.TILE_SIZE * 2 - self.rect.height) / 2],
            [12 * self.level.TILE_SIZE + (self.level.TILE_SIZE * 2 - self.rect.width) / 2, (self.level.TILE_SIZE * 2 - self.rect.height) / 2],
            [24 * self.level.TILE_SIZE + (self.level.TILE_SIZE * 2 - self.rect.width) / 2,  (self.level.TILE_SIZE * 2 - self.rect.height) / 2]
        ]

        random.shuffle(available_positions)

        for pos in available_positions:

            enemy_rect = pygame.Rect(pos, [26, 26])

            # collisions with other enemies
            collision = False
            for enemy in enemies:
                if enemy_rect.colliderect(enemy.rect):
                    collision = True
                    continue

            if collision:
                continue

            # collisions with players
            collision = False
            for player in players:
                if enemy_rect.colliderect(player.rect):
                    collision = True
                    continue

            if collision:
                continue

            return pos
        return False

    def move(self):
        """ move enemy if possible """

        global players, enemies, bonuses

        if self.state != self.STATE_ALIVE or self.paused or self.paralised:
            return

        if self.path == []:
            self.path = self.generatePath(None, True)

        new_position = self.path.pop(0)

        # move enemy
        if self.direction == self.DIR_UP:
            if new_position[1] < 0:
                self.path = self.generatePath(self.direction, True)
                return
        elif self.direction == self.DIR_RIGHT:
            if new_position[0] > (208 - 13):
                self.path = self.generatePath(self.direction, True)
                return
        elif self.direction == self.DIR_DOWN:
            if new_position[1] > (208- 13):
                self.path = self.generatePath(self.direction, True)
                return
        elif self.direction == self.DIR_LEFT:
            if new_position[0] < 0:
                self.path = self.generatePath(self.direction, True)
                return

        new_rect = pygame.Rect(new_position, [13, 13])

        # collisions with tiles
        if new_rect.collidelist(self.level.obstacle_rects) != -1:
            self.path = self.generatePath(self.direction, True)
            return

        # collisions with other enemies
        for enemy in enemies:
            # if enemy != self:
            #     if abs(self.rect.centerx - enemy.rect.centerx) < 30 and abs(self.rect.centery-enemy.rect.centery) < 30:
            #         self.rect.centerx += 50
            if enemy != self and new_rect.colliderect(enemy.rect):
                self.turnAround()
                self.path = self.generatePath(self.direction)
                return

        # collisions with players
        for player in players:
            # if abs(self.rect.centerx - player.rect.centerx) < 30 and abs(self.rect.centery-player.rect.centery) < 30:
            #     self.rect.centerx += 50
            # if new_rect.colliderect(player.rect):
            #     self.turnAround()
            #     self.path = self.generatePath(self.direction)
                # self.rect.topleft = new_rect.topleft
                # return
            pass

        # collisions with bonuses
        for bonus in bonuses:
            if new_rect.colliderect(bonus.rect):
                bonuses.remove(bonus)

        # if no collision, move enemy
        self.rect.topleft = new_rect.topleft


    def update(self, time_passed):
        Tank.update(self, time_passed)
        if self.state == self.STATE_ALIVE and not self.paused:
            self.move()

    def generatePath(self, direction = None, fix_direction = False):
        """ If direction is specified, try continue that way, otherwise choose at random
        """

        all_directions = [self.DIR_UP, self.DIR_RIGHT, self.DIR_DOWN, self.DIR_LEFT]

        if direction == None:
            if self.direction in [self.DIR_UP, self.DIR_RIGHT]:
                opposite_direction = self.direction + 2
            else:
                opposite_direction = self.direction - 2
            directions = all_directions
            random.shuffle(directions)
            directions.remove(opposite_direction)
            directions.append(opposite_direction)
        else:
            if direction in [self.DIR_UP, self.DIR_RIGHT]:
                opposite_direction = direction + 2
            else:
                opposite_direction = direction - 2

            if direction in [self.DIR_UP, self.DIR_RIGHT]:
                opposite_direction = direction + 2
            else:
                opposite_direction = direction - 2
            directions = all_directions
            random.shuffle(directions)
            directions.remove(opposite_direction)
            directions.remove(direction)
            directions.insert(0, direction)
            directions.append(opposite_direction)

        # at first, work with general units (steps) not px
        x = int(round(self.rect.left / 8))
        y = int(round(self.rect.top / 8))

        new_direction = None

        for direction in directions:
            if direction == self.DIR_UP and y > 1:
                new_pos_rect = self.rect.move(0, -8)
                if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
                    new_direction = direction
                    break
            elif direction == self.DIR_RIGHT and x < 24:
                new_pos_rect = self.rect.move(8, 0)
                if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
                    new_direction = direction
                    break
            elif direction == self.DIR_DOWN and y < 24:
                new_pos_rect = self.rect.move(0, 8)
                if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
                    new_direction = direction
                    break
            elif direction == self.DIR_LEFT and x > 1:
                new_pos_rect = self.rect.move(-8, 0)
                if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
                    new_direction = direction
                    break

        # if we can go anywhere else, turn around
        if new_direction == None:
            new_direction = opposite_direction

        # fix tanks position
        if fix_direction and new_direction == self.direction:
            fix_direction = False

        self.rotate(new_direction, fix_direction)

        positions = []

        x = self.rect.left
        y = self.rect.top

        if new_direction in (self.DIR_RIGHT, self.DIR_LEFT):
            axis_fix = self.nearest(y, 8) - y
        else:
            axis_fix = self.nearest(x, 8) - x
        axis_fix = 0

        pixels = self.nearest(random.randint(1, 12) * 16, 16) + axis_fix + 3

        if new_direction == self.DIR_UP:
            for px in range(0, pixels, self.speed):
                positions.append([x, y-px])
        elif new_direction == self.DIR_RIGHT:
            for px in range(0, pixels, self.speed):
                positions.append([x+px, y])
        elif new_direction == self.DIR_DOWN:
            for px in range(0, pixels, self.speed):
                positions.append([x, y+px])
        elif new_direction == self.DIR_LEFT:
            for px in range(0, pixels, self.speed):
                positions.append([x-px, y])

        return positions



class Player(Tank):

    def __init__(self, level, type, position = None, direction = None, filename = None):

        Tank.__init__(self, level, type, position = None, direction = None, filename = None)

        global sprites

        if filename == None:
            filename = (0, 0, 16, 16)

        self.start_position = position
        self.start_direction = direction

        self.lives = 3

        # total score
        self.score = 0

        # store how many bonuses in this stage this player has collected
        self.trophies = {
            "bonus" : 0,
            "enemy0" : 0,
            "enemy1" : 0,
            "enemy2" : 0,
            "enemy3" : 0
        }

        self.image = sprites.subsurface(filename)
        self.image_up = self.image;
        self.image_left = pygame.transform.rotate(self.image, 90)
        self.image_down = pygame.transform.rotate(self.image, 180)
        self.image_right = pygame.transform.rotate(self.image, 270)

        if direction == None:
            self.rotate(self.DIR_UP, False)
        else:
            self.rotate(direction, False)


    def move(self, direction):
        """ move player if possible """

        global players, enemies, bonuses

        if self.state == self.STATE_EXPLODING:
            if not self.explosion.active:
                self.state = self.STATE_DEAD
                del self.explosion

        if self.state != self.STATE_ALIVE:
            return

        # rotate player
        if self.direction != direction:
            self.rotate(direction)

        if self.paralised:
            return

        # move player
        new_position = [self.rect.left, self.rect.top]
        if direction == self.DIR_UP:
            new_position = [self.rect.left, self.rect.top - self.speed]
            if new_position[1] < 0:
                return
        elif direction == self.DIR_RIGHT:
            new_position = [self.rect.left + self.speed, self.rect.top]
            if new_position[0] > (208 - 13):
                return
        elif direction == self.DIR_DOWN:
            new_position = [self.rect.left, self.rect.top + self.speed]
            if new_position[1] > (208 - 13):
                return
        elif direction == self.DIR_LEFT:
            new_position = [self.rect.left - self.speed, self.rect.top]
            if new_position[0] < 0:
                return

        player_rect = pygame.Rect(new_position, [13, 13])

        # collisions with tiles
        if player_rect.collidelist(self.level.obstacle_rects) != -1:
            return

        # collisions with other players
        for player in players:
            if player != self and player.state == player.STATE_ALIVE and player_rect.colliderect(player.rect) == True:
                return

        # collisions with enemies
        for enemy in enemies:
            if player_rect.colliderect(enemy.rect) == True:
                return

        # collisions with bonuses
        for bonus in bonuses:
            if player_rect.colliderect(bonus.rect) == True:
                self.bonus = bonus

        #if no collision, move player
        self.rect.topleft = (new_position[0], new_position[1])

    def reset(self):
        """ reset player """
        self.rotate(self.start_direction, False)
        pos = [
            random.uniform(self.level.TILE_SIZE, 12*self.level.TILE_SIZE),
            random.uniform(self.level.TILE_SIZE, 12*self.level.TILE_SIZE)
        ]
        # self.rect.topleft = self.start_position
        self.rect.topleft = pos
        self.superpowers = 0
        self.max_active_bullets = 1
        self.health = 100
        self.paralised = False
        self.paused = False
        self.pressed = [False] * 4
        self.state = self.STATE_ALIVE

class Game:

    # direction constants
    (DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT) = range(4)

    TILE_SIZE = 16

    def __init__(self):

        global screen, sprites, play_sounds, sounds

        # center window
        os.environ['SDL_VIDEO_WINDOW_POS'] = 'center'

        if play_sounds:
            pygame.mixer.pre_init(44100, -16, 1, 512)

        pygame.init()


        pygame.display.set_caption("Battle City")

        # size = width, height = 480, 416
        size = width, height = 240, 208

        '''
        if "-f" in sys.argv[1:]:
            screen = pygame.display.set_mode(size, pygame.FULLSCREEN)
        else:
        '''
        screen = pygame.display.set_mode(size)

        self.clock = pygame.time.Clock()

        # load sprites (funky version)
        #sprites = pygame.transform.scale2x(pygame.image.load("images/sprites.gif"))
        # load sprites (pixely version)
        # sprites = pygame.transform.scale(pygame.image.load("images/sprites.gif"), [192, 224])
        sprites = pygame.transform.scale(pygame.image.load("images/sprites.gif"), [96, 112])
        #screen.set_colorkey((0,138,104))

        # pygame.display.set_icon(sprites.subsurface(0, 0, 13*2, 13*2))
        pygame.display.set_icon(sprites.subsurface(0, 0, 13, 13))

        # load sounds
        if play_sounds:
            pygame.mixer.init(44100, -16, 1, 512)

            sounds["start"] = pygame.mixer.Sound("sounds/gamestart.ogg")
            sounds["end"] = pygame.mixer.Sound("sounds/gameover.ogg")
            sounds["score"] = pygame.mixer.Sound("sounds/score.ogg")
            sounds["bg"] = pygame.mixer.Sound("sounds/background.ogg")
            sounds["fire"] = pygame.mixer.Sound("sounds/fire.ogg")
            sounds["bonus"] = pygame.mixer.Sound("sounds/bonus.ogg")
            sounds["explosion"] = pygame.mixer.Sound("sounds/explosion.ogg")
            sounds["brick"] = pygame.mixer.Sound("sounds/brick.ogg")
            sounds["steel"] = pygame.mixer.Sound("sounds/steel.ogg")

        self.enemy_life_image = sprites.subsurface(81, 57, 7, 7)
        self.player_life_image = sprites.subsurface(89, 56, 7, 8)
        self.flag_image = sprites.subsurface(64, 49, 16, 15)

        # this is used in intro screen
        self.player_image = pygame.transform.rotate(sprites.subsurface(0, 0, 13, 13), 270)

        # if true, no new enemies will be spawn during this time
        self.timefreeze = False

        # load custom font
        self.font = pygame.font.Font("fonts/prstart.ttf", 16)

        # pre-render game over text
        self.im_game_over = pygame.Surface((64, 40))
        self.im_game_over.set_colorkey((0,0,0))
        self.im_game_over.blit(self.font.render("GAME", False, (127, 64, 64)), [0, 0])
        self.im_game_over.blit(self.font.render("OVER", False, (127, 64, 64)), [0, 20])
        self.game_over_y = 416+40

        # number of players. here is defined preselected menu value
        self.nr_of_players = 1

        del players[:]
        del bullets[:]
        del enemies[:]
        del bonuses[:]


    def triggerBonus(self, bonus, player):
        """ Execute bonus powers """

        global enemies, labels, play_sounds, sounds

        if play_sounds:
            sounds["bonus"].play()

        player.trophies["bonus"] += 1
        player.score += 500

        if bonus.bonus == bonus.BONUS_GRENADE:
            for enemy in enemies:
                enemy.explode()
        elif bonus.bonus == bonus.BONUS_HELMET:
            self.shieldPlayer(player, True, 10000)
        elif bonus.bonus == bonus.BONUS_SHOVEL:
            self.level.buildFortress(self.level.TILE_STEEL)
            gtimer.add(10000/options["game_speed"], lambda :self.level.buildFortress(self.level.TILE_BRICK), 1)
        elif bonus.bonus == bonus.BONUS_STAR:
            player.superpowers += 1
            if player.superpowers == 2:
                player.max_active_bullets = 2
        elif bonus.bonus == bonus.BONUS_TANK:
            player.lives += 1
        elif bonus.bonus == bonus.BONUS_TIMER:
            self.toggleEnemyFreeze(True)
            gtimer.add(10000/options["game_speed"], lambda :self.toggleEnemyFreeze(False), 1)
        bonuses.remove(bonus)

        labels.append(Label(bonus.rect.topleft, "500", 500))

    def shieldPlayer(self, player, shield = True, duration = None):
        """ Add/remove shield
        player: player (not enemy)
        shield: true/false
        duration: in ms. if none, do not remove shield automatically
        """
        player.shielded = shield
        if shield:
            player.timer_uuid_shield = gtimer.add(100/options["game_speed"], lambda :player.toggleShieldImage())
        else:
            gtimer.destroy(player.timer_uuid_shield)

        if shield and duration != None:
            gtimer.add(duration/options["game_speed"], lambda :self.shieldPlayer(player, False), 1)


    def spawnEnemy(self):
        """ Spawn new enemy if needed
        Only add enemy if:
            - there are at least one in queue
            - map capacity hasn't exceeded its quota
            - now isn't timefreeze
        """

        global enemies

        if len(enemies) >= self.level.max_active_enemies:
            return
        if len(self.level.enemies_left) < 1 or self.timefreeze:
            return
        enemy = Enemy(self.level, 1)

        enemies.append(enemy)


    def respawnPlayer(self, player, clear_scores = False):
        """ Respawn player """
        player.reset()

        if clear_scores:
            player.trophies = {
                "bonus" : 0, "enemy0" : 0, "enemy1" : 0, "enemy2" : 0, "enemy3" : 0
            }

        # self.shieldPlayer(player, True, 4000)


    def showMenu(self):
        """ Show game menu
        Redraw screen only when up or down key is pressed. When enter is pressed,
        exit from this screen and start the game with selected number of players
        """

        global players, screen

        # stop game main loop (if any)
        self.running = False

        # clear all timers
        del gtimer.timers[:]

        # set current stage to 0
        self.stage = 0

        del players[:]
        self.nextLevel()

    def reloadPlayers(self):
        """ Init players
        If players already exist, just reset them
        """

        global players

        if len(players) == 0:
            # first player
            x = 4 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
            y = 12 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2

            player = Player(
                # self.level, 0, [x, y], self.DIR_UP, (0, 0, 13*2, 13*2)
                self.level, 0, [x, y], self.DIR_UP, (0, 0, 13, 13)
            )
            players.append(player)

            # second player
            if self.nr_of_players == 2:
                x = 8 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
                y = 12 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
                player = Player(
                    self.level, 0, [x, y], self.DIR_UP, (16, 0, 13, 13)
                )
                player.controls = [102, 119, 100, 115, 97]
                players.append(player)

        for player in players:
            player.level = self.level
            self.respawnPlayer(player, True)


    def showScores(self):
        ''' Directly go to next level '''
        # stop game main loop (if any)
        self.running = False
        # clear all timers
        del gtimer.timers[:]
        if self.game_over:
            self.gameOverScreen()
        else:
            self.nextLevel()
        return


    def draw(self):
        global screen, castle, players, enemies, bullets, bonuses

        screen.fill([0, 0, 0])

        self.level.draw([self.level.TILE_EMPTY, self.level.TILE_BRICK, self.level.TILE_STEEL, self.level.TILE_FROZE, self.level.TILE_WATER])

        castle.draw()

        for enemy in enemies:
            enemy.draw()

        for label in labels:
            label.draw()

        for player in players:
            player.draw()

        for bullet in bullets:
            bullet.draw()

        for bonus in bonuses:
            bonus.draw()

        self.level.draw([self.level.TILE_GRASS])

        if self.game_over:
            if self.game_over_y > 188:
                self.game_over_y -= 4
            screen.blit(self.im_game_over, [176, self.game_over_y]) # 176=(416-64)/2

        self.drawSidebar()

        pygame.display.flip()

    def drawSidebar(self):

        global screen, players, enemies

        x = 416
        y = 0
        screen.fill([100, 100, 100], pygame.Rect([416, 0], [64, 416]))

        xpos = x + 16
        ypos = y + 16

        # draw enemy lives
        for n in range(len(self.level.enemies_left) + len(enemies)):
            screen.blit(self.enemy_life_image, [xpos, ypos])
            if n % 2 == 1:
                xpos = x + 16
                ypos+= 17
            else:
                xpos += 17

        # players' lives
        if pygame.font.get_init():
            text_color = pygame.Color('black')
            for n in range(len(players)):
                if n == 0:
                    screen.blit(self.font.render(str(n+1)+"P", False, text_color), [x+16, y+200])
                    screen.blit(self.font.render(str(players[n].lives), False, text_color), [x+31, y+215])
                    screen.blit(self.player_life_image, [x+17, y+215])
                else:
                    screen.blit(self.font.render(str(n+1)+"P", False, text_color), [x+16, y+240])
                    screen.blit(self.font.render(str(players[n].lives), False, text_color), [x+31, y+255])
                    screen.blit(self.player_life_image, [x+17, y+255])

            screen.blit(self.flag_image, [x+17, y+280])
            screen.blit(self.font.render(str(self.stage), False, text_color), [x+17, y+312])


    def drawIntroScreen(self, put_on_surface = True):
        """ Draw intro (menu) screen
        @param boolean put_on_surface If True, flip display after drawing
        @return None
        """

        global screen

        screen.fill([0, 0, 0])

        if pygame.font.get_init():

            hiscore = self.loadHiscore()

            screen.blit(self.font.render("HI- "+str(hiscore), True, pygame.Color('white')), [170, 35])

            screen.blit(self.font.render("1 PLAYER", True, pygame.Color('white')), [165, 250])
            screen.blit(self.font.render("2 PLAYERS", True, pygame.Color('white')), [165, 275])

            screen.blit(self.font.render("(c) 1980 1985 NAMCO LTD.", True, pygame.Color('white')), [50, 350])
            screen.blit(self.font.render("ALL RIGHTS RESERVED", True, pygame.Color('white')), [85, 380])


        if self.nr_of_players == 1:
            screen.blit(self.player_image, [125, 245])
        elif self.nr_of_players == 2:
            screen.blit(self.player_image, [125, 270])

        self.writeInBricks("battle", [65, 80])
        self.writeInBricks("city", [129, 160])

        if put_on_surface:
            pygame.display.flip()

    def animateIntroScreen(self):
        """ Slide intro (menu) screen from bottom to top
        If Enter key is pressed, finish animation immediately
        @return None
        """

        global screen

        self.drawIntroScreen(False)
        screen_cp = screen.copy()

        screen.fill([0, 0, 0])

        y = 416
        while y > 0:
            time_passed = self.clock.tick(50)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        y = 0
                        break

            screen.blit(screen_cp, [0, y])
            pygame.display.flip()
            y -= 5

        screen.blit(screen_cp, [0, 0])
        pygame.display.flip()


    def chunks(self, l, n):
        """ Split text string in chunks of specified size
        @param string l Input string
        @param int n Size (number of characters) of each chunk
        @return list
        """
        return [l[i:i+n] for i in range(0, len(l), n)]

    def writeInBricks(self, text, pos):
        """ Write specified text in "brick font"
        Only those letters are available that form words "Battle City" and "Game Over"
        Both lowercase and uppercase are valid input, but output is always uppercase
        Each letter consists of 7x7 bricks which is converted into 49 character long string
        of 1's and 0's which in turn is then converted into hex to save some bytes
        @return None
        """

        global screen, sprites

        bricks = sprites.subsurface(56, 64, 8, 8)
        brick1 = bricks.subsurface((0, 0, 8, 8))
        brick2 = bricks.subsurface((8, 0, 8, 8))
        brick3 = bricks.subsurface((8, 8, 8, 8))
        brick4 = bricks.subsurface((0, 8, 8, 8))

        alphabet = {
            "a" : "0071b63c7ff1e3",
            "b" : "01fb1e3fd8f1fe",
            "c" : "00799e0c18199e",
            "e" : "01fb060f98307e",
            "g" : "007d860cf8d99f",
            "i" : "01f8c183060c7e",
            "l" : "0183060c18307e",
            "m" : "018fbffffaf1e3",
            "o" : "00fb1e3c78f1be",
            "r" : "01fb1e3cff3767",
            "t" : "01f8c183060c18",
            "v" : "018f1e3eef8e08",
            "y" : "019b3667860c18"
        }

        abs_x, abs_y = pos

        for letter in text.lower():

            binstr = ""
            for h in self.chunks(alphabet[letter], 2):
                binstr += str(bin(int(h, 16)))[2:].rjust(8, "0")
            binstr = binstr[7:]

            x, y = 0, 0
            letter_w = 0
            surf_letter = pygame.Surface((56, 56))
            for j, row in enumerate(self.chunks(binstr, 7)):
                for i, bit in enumerate(row):
                    if bit == "1":
                        if i%2 == 0 and j%2 == 0:
                            surf_letter.blit(brick1, [x, y])
                        elif i%2 == 1 and j%2 == 0:
                            surf_letter.blit(brick2, [x, y])
                        elif i%2 == 1 and j%2 == 1:
                            surf_letter.blit(brick3, [x, y])
                        elif i%2 == 0 and j%2 == 1:
                            surf_letter.blit(brick4, [x, y])
                        if x > letter_w:
                            letter_w = x
                    x += 8
                x = 0
                y += 8
            screen.blit(surf_letter, [abs_x, abs_y])
            abs_x += letter_w + 16

    def toggleEnemyFreeze(self, freeze = True):
        """ Freeze/defreeze all enemies """

        global enemies

        for enemy in enemies:
            enemy.paused = freeze
        self.timefreeze = freeze


    def loadHiscore(self):
        """ Load hiscore
        Really primitive version =] If for some reason hiscore cannot be loaded, return 20000
        @return int
        """
        filename = ".hiscore"
        if not os.path.isfile(filename):
            return 20000

        f = open(filename, "r")
        hiscore = int(f.read())

        if hiscore > 19999 and hiscore < 1000000:
            return hiscore
        else:
            return 20000

    def saveHiscore(self, hiscore):
        """ Save hiscore
        @return boolean
        """
        try:
            f = open(".hiscore", "w")
        except:
            return False
        f.write(str(hiscore))
        f.close()
        return True


    def finishLevel(self):
        """ Finish current level
        Show earned scores and advance to the next stage
        """

        global play_sounds, sounds

        if play_sounds:
            sounds["bg"].stop()

        self.active = False
        gtimer.add(3000/options["game_speed"], lambda :self.showScores(), 1)


    def nextLevel(self):
        """ Start next level """

        global castle, players, bullets, bonuses, play_sounds, sounds, episodes

        del bullets[:]
        del enemies[:]
        del bonuses[:]
        castle.rebuild()
        del gtimer.timers[:]

        # if episodes >= options.train_episodes - 10:
             # options["game_speed"] = 2.0

        # load level
        #self.stage += 1
        self.stage = 1      # stick on 1st stage

        self.level = Level(self.stage)
        self.timefreeze = False

        # set number of enemies by types (basic, fast, power, armor) according to level
        with open("enemies/moderate_basic") as f:
            lines = f.read().splitlines()
        levels_enemies = []
        for line in lines:
            line_int = []
            for str in line.split():
                line_int.append(int(str))
            levels_enemies.append(line_int)

        if self.stage <= 35:
            enemies_l = levels_enemies[self.stage - 1]
        else:
            enemies_l = levels_enemies[34]

        self.level.enemies_left = [0]*enemies_l[0] + [1]*enemies_l[1] + [2]*enemies_l[2] + [3]*enemies_l[3]
        random.shuffle(self.level.enemies_left)

        if play_sounds:
            sounds["start"].play()
            gtimer.add(4330, lambda :sounds["bg"].play(-1), 1)

        self.reloadPlayers()

        gtimer.add(0, lambda :self.spawnEnemy())

        # if True, start "game over" animation
        self.game_over = False

        # if False, game will end w/o "game over" bussiness
        self.running = True

        # if False, players won't be able to do anything
        self.active = True



class Env:
    def __init__(self):
        global options
        options = dict()
        options["level_type"] = 'minimal'
        options["game_speed"] = 50
        options["train_episodes"] = 20
        options["draw"] = True
        self.game = None
        self.lastScore = 0

    def reset(self):
        global gtimer, sprites, screen, players, enemies, bullets, bonuses, labels, play_sounds, sounds, castle
        gtimer = Timer()

        sprites = None
        screen = None
        players = []
        enemies = []
        bullets = []
        bonuses = []
        labels = []

        play_sounds = False
        sounds = {}

        self.game = Game()
        castle = Castle()
        self.game.showMenu()

        player = players[0]

        # Modify this line to change the state representation of the game
        # game_state = state.CompleteState(player, enemies, bullets, self.game.level.mapr, castle, bonuses)
        # img = pygame.surfarray.array2d(screen)
        # game_state = sp.misc.imresize(img[:416, :416], (104, 104))
        return state.getRawState(screen)

    '''
    @returns state, reward, gameover
    '''
    def step_return(self, episode_over):
        player = players[0]

        # game_state = state.CompleteState(player, enemies, bullets, self.game.level.mapr, castle, bonuses)
        game_state = state.getRawState(screen)

        time_reward = 1
        reward = player.score + time_reward
        return game_state , reward, episode_over, None
        # return DangerExtractor.getFeatures(game_state) , reward, episode_over, None

    def render(self):
        self.game.draw()


    '''
    @returns state, reward, episode_over
    '''
    def step(self, action):

        # Main game loop

        time_passed = self.game.clock.tick(50*options["game_speed"])

        player = players[0]
        # Get action from learning agent
        # The complete state includes: players, enemies, bullets, level, castle, bonus

        if player.state == player.STATE_ALIVE and not self.game.game_over and self.game.active:
            if action == Const.DO_FIRE:
                # player.fire()
                pass
            else:
                player.move(action-1)

        player.update(time_passed)
        player.score = 0

        for enemy in enemies:
            if enemy.state == enemy.STATE_DEAD and not self.game.game_over and self.game.active:
                enemies.remove(enemy)
                if len(self.game.level.enemies_left) == 0 and len(enemies) == 0:
                    print("Win")
                    return self.step_return(True)  #episode over
            else:
                enemy.update(time_passed)

        if not self.game.game_over and self.game.active:
            if player.state == player.STATE_DEAD:
                self.game.superpowers = 0
                player.lives -= 1
                player.score -= 1000
                if player.lives > 0:
                    self.game.respawnPlayer(player)
                else:
                    return self.step_return(True)      #episode over

        for bullet in bullets:
            if bullet.state == bullet.STATE_REMOVED:
                bullets.remove(bullet)
            else:
                bullet.update()

        for label in labels:
            if not label.active:
                labels.remove(label)

        if not self.game.game_over:
            if not castle.active:
                print("Castle Dead")
                return self.step_return(True)      #episode over

        gtimer.update(time_passed)


        return self.step_return(False)


if __name__ == "__main__":
    env = Env()
    env.reset()

    for i in range(0, 100):
        env.step(0)
        env.render()
        print(i)

    while True:
        array = pygame.surfarray.array2d(screen)
        array = array[:208, :208]
        array = sp.misc.imresize(array, (104, 104))
        print(array.shape)
        img = smp.toimage(array)
        img.show()
        while True:
            pass
        break
