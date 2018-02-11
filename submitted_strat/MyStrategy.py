from model.ActionType import ActionType
from model.VehicleType import VehicleType
from model.Game import Game
from model.Move import Move
from model.Player import Player
from model.World import World

import time
import math
import random

import numpy as np
from scipy.cluster.vq import kmeans2 as kmeans
import scipy.cluster.hierarchy as hac

all_vehicle_types = np.linspace(0, 4, 5, dtype=int)


class MyStrategy:
    AERIAL_SCALE_FACTOR = 1.4
    GROUND_SCALE_FACTOR = 2.1
    BASE_SQUAD_SIZE = 54
    AERIAL_SQUAD_SIZE = BASE_SQUAD_SIZE * AERIAL_SCALE_FACTOR
    GROUND_SQUAD_SIZE = BASE_SQUAD_SIZE * GROUND_SCALE_FACTOR
    DEF_AERIAL_WAIT = 25
    EPS = 0.01

    SIG_NUKE_DAMAGE_RATIO = 0.25
    SIG_POSITION_DIST = 32  # 45 = 250/18 (one second of moving)

    VEHICLE_TYPE_NAMES = ['ARRV', 'FIGHTER', 'HELICOPTER', 'IFV', 'TANK']

    all_vehicles = {}

    aerial_formation_type = ['right_down', 'down_right']
    aerial_formation_stage = 0
    ground_formation_stage = 0

    lock = 'free'
    enemy_is_nuking = False
    en_nuke_tick = -1

    def move(self, me: Player, world: World, game: Game, move: Move):
        self.update(me, world, game)
        self.clear_cache()

        # PREPROCESSING OF MEDIANS
        #  me
        median_me_all = self.get_median(me)
        median_me_ground = self.get_median(me, types=[0, 3, 4])
        median_me_aerial = self.get_median(me, types=[1, 2])
        median_me_of_type = [self.get_median(me, types=[type])
                             for type in range(5)]
        # enemy
        median_en_all = self.get_median(self.enemy)
        # median_en_ground = self.get_median(self.enemy, types=[0, 3, 4])
        # median_en_aerial = self.get_median(self.enemy, types=[1, 2])
        # median_en_of_type = [self.get_median(self.enemy, types=[type])
        #                      for type in range(5)]

        if world.tick_index % 100 == 0:
            print("NOW AT TICK", world.tick_index)
            print(median_me_all)
            print("CENTERS:")
            for type in range(5):
                median = median_me_of_type[type]
                print(" ", self.VEHICLE_TYPE_NAMES[type], median)
                X = np.array(list(self.get_data_of_units(me, types=[type])[0]))
                Y = np.array(list(self.get_data_of_units(me, types=[type])[1]))
                print(" ", " ", "domain:", X.min(), X.max())
                print(" ", " ", "range:", Y.min(), Y.max())
            print()

        # INIT INFO
        if world.tick_index == 0:
            fighter_x, fighter_y = median_me_of_type[1]
            helicopter_x, helicopter_y = median_me_of_type[2]
            if (fighter_x == helicopter_x and fighter_y < helicopter_y) or \
               (fighter_x > fighter_y and not (fighter_x < helicopter_x and
                                               fighter_y == helicopter_y)):
                self.aerial_formation_type = 'right_down'
            else:
                self.aerial_formation_type = 'down_right'

            self.next_formation_stage(stage='aerial')
            self.next_formation_stage(stage='ground')

        # PLANNED TACTICS FOR ROUND II
        # 1) do phase 1 improved using potential flows
        # 2) split into 4, treat these groups as single entities
        # 3) use potential flows to occupy facilities
        # 4) produce enough units for a 5th group
        # 5) if one facility is not occupied, then use the 5th group to occupy it 
        #    else 2 or more groups to invade an facility and let the 5th remain on it

        # PHASE I: FORMATION
        ## AERIAL UNITS
        if self.aerial_formation_stage == 1 and self.lock == 'free':      # select fighter planes
            self.act_new_select(move, vehicle_type=1, stage='aerial')
            self.lock = 'aerial'
            return
        elif self.aerial_formation_stage == 2 and self.lock == 'aerial':  # move fighter planes
            if self.aerial_formation_type == 'right_down':
                self.act_move(move, source=median_me_of_type[1],
                                    dest=(270, median_me_of_type[1][1]), stage='aerial')
            else:
                self.act_move(move, source=median_me_of_type[1],
                                    dest=(median_me_of_type[1][0], 270), stage='aerial')
            self.aerial_wait = self.DEF_AERIAL_WAIT
            self.lock = 'free'
            return
        elif self.aerial_formation_stage == 3:                            # wait for fighter planes to be in position
            fighter_x, fighter_y = median_me_of_type[1]
            if self.aerial_formation_type == 'right_down':
                if abs(fighter_x - 270) < self.EPS:
                    if self.aerial_wait == 0:
                        self.next_formation_stage(stage='aerial')
                    else:
                        self.aerial_wait -= 1
            else:
                if abs(fighter_y - 270) < self.EPS:
                    if self.aerial_wait == 0:
                        self.next_formation_stage(stage='aerial')
                    else:
                        self.aerial_wait -= 1
        elif self.aerial_formation_stage == 4 and self.lock == 'free':      # select fighter planes
            self.act_new_select(move, vehicle_type=1, stage='aerial')
            self.lock = 'aerial'
            return
        elif self.aerial_formation_stage == 5 and self.lock == 'aerial':  # move fighter planes
            if self.aerial_formation_type == 'right_down':
                self.act_move(move, source=median_me_of_type[1],
                                    dest=(median_me_of_type[1][0], 270), stage='aerial')
            else:
                self.act_move(move, source=median_me_of_type[1],
                                    dest=(270, median_me_of_type[1][1]), stage='aerial')
            self.lock = 'free'
            return
        elif self.aerial_formation_stage == 6 and self.lock == 'free':    # select helicopters
            self.act_new_select(move, vehicle_type=2, stage='aerial')
            self.lock = 'aerial'
            return
        elif self.aerial_formation_stage == 7 and self.lock == 'aerial':  # move helicopters
            if self.aerial_formation_type == 'right_down':
                dest = (274 - self.AERIAL_SQUAD_SIZE - 20, 274)
            else:
                dest = (274, 274 - self.AERIAL_SQUAD_SIZE - 20)
            self.act_move(move, source=median_me_of_type[2], dest=dest, stage='aerial')
            self.aerial_wait = self.DEF_AERIAL_WAIT
            self.lock = 'free'
            return
        elif self.aerial_formation_stage == 8:                            # wait for both to be in position
            if not self.check_in_pos(median_me_of_type[1], (270, 270)):
                pass
            elif (self.aerial_formation_type == 'right_down' and
                  self.check_in_pos(median_me_of_type[2], (274 - self.AERIAL_SQUAD_SIZE - 20, 274))) or \
                 (self.aerial_formation_type == 'down_right' and
                  self.check_in_pos(median_me_of_type[2], (274, 274 - self.AERIAL_SQUAD_SIZE - 20))):
                    if self.aerial_wait == 0:
                        self.next_formation_stage(stage='aerial')
                    else:
                        self.aerial_wait -= 1
        elif self.aerial_formation_stage == 9 and self.lock == 'free':    # select fighter planes
            self.act_new_select(move, vehicle_type=1, stage='aerial')
            self.lock = 'aerial'
            return
        elif self.aerial_formation_stage == 10 and self.lock == 'aerial':  # scale fighter planes
            self.act_scale(move, median_me_of_type[1], stage='aerial')
            self.lock = 'free'
            return
        elif self.aerial_formation_stage == 11 and self.lock == 'free':   # select helicopters
            self.act_new_select(move, vehicle_type=2, stage='aerial')
            self.lock = 'aerial'
            return
        elif self.aerial_formation_stage == 12 and self.lock == 'aerial': # scale helicopters
            self.act_scale(move, median_me_of_type[2], stage='aerial')
            self.aerial_wait = self.DEF_AERIAL_WAIT
            return
        elif self.aerial_formation_stage == 13:                           # wait for both to scale
            if self.check_if_scaled(vehicle_type=1, target_size=self.AERIAL_SQUAD_SIZE) and \
               self.check_if_scaled(vehicle_type=2, target_size=self.AERIAL_SQUAD_SIZE):
                if self.aerial_wait == 0:
                    self.next_formation_stage(stage='aerial')
                else:
                    self.aerial_wait -= 1
        elif self.aerial_formation_stage == 14 and self.lock == 'aerial': # finish formation
            self.act_move(move, source=median_me_of_type[2],
                                dest=(274, 274), stage='aerial')
            self.aerial_wait = self.DEF_AERIAL_WAIT
            return
        elif self.aerial_formation_stage == 15:                           # wait for helicopters to be in position
            if self.check_in_pos(median_me_of_type[2], (274, 274)):
                if self.aerial_wait == 0:
                    self.next_formation_stage(stage='aerial')
                else:
                    self.aerial_wait -= 1
        elif self.aerial_formation_stage == 16 and self.lock == 'aerial': # add fighters to selection
            self.act_add_select(move, vehicle_type=1, stage='aerial')
            return
        elif self.aerial_formation_stage == 17 and self.lock == 'aerial': # move to centroid of ground units
            self.act_move(move, source=median_me_of_type[2],
                                dest=self.get_centroid(self.me, types=[0, 3, 4]), stage='aerial')
            self.lock = 'free'
            return
        ## GROUND UNITS
        '''move_to = {45:56.7, 119:178.3, 193:299.9}
        if self.ground_formation_stage in [1, 2, 3, 4, 5, 6]:
            i = (self.ground_formation_stage-1)//2
            vehicle_type = [0, 3, 4][i]
            if self.ground_formation_stage % 2 == 1 and self.lock == 'free':
                self.act_new_select(move, vehicle_type, stage='ground')
                self.lock = 'ground'
            elif self.lock == 'ground':
                med = median_me_of_type[vehicle_type]
                self.act_move(move, source=med,
                                    dest=(move_to[int(med[0])]+4.1*i,
                                          move_to[int(med[1])]+4.1*i), stage='ground')
                self.lock = 'free'
            return
        elif world.tick_index > 400 and self.ground_formation_stage in [7, 8, 9, 10, 11, 12]:
            i = (self.ground_formation_stage-7)//2
            vehicle_type = [0, 3, 4][i]
            if self.ground_formation_stage % 2 == 1 and self.lock == 'free':
                self.act_new_select(move, vehicle_type, stage='ground')
                self.lock = 'ground'
            elif self.lock == 'ground':
                med = median_me_of_type[vehicle_type]
                self.act_scale(move, pivot=med, factor=self.GROUND_SCALE_FACTOR, stage='ground')
                self.lock = 'free'
            return
        elif world.tick_index > 700 and self.ground_formation_stage in [13, 14, 15, 16, 17, 18]:
            i = (self.ground_formation_stage-13)//2
            vehicle_type = [0, 3, 4][i]
            if self.ground_formation_stage % 2 == 1 and self.lock == 'free':
                self.act_new_select(move, vehicle_type, stage='ground')
                self.lock = 'ground'
            elif self.lock == 'ground':
                med = median_me_of_type[vehicle_type]
                self.act_move(move, relative=(median_me_ground[0]-med[0], 0), stage='ground')
                self.lock = 'free'
        '''
        if self.ground_formation_stage == 1 and self.lock == 'free':      # select ARRVs
            self.act_new_select(move, vehicle_type=0, stage='ground')
            self.lock = 'ground'
            return
        elif self.ground_formation_stage == 2 and self.lock == 'ground':  # add IFVs to selection
            self.act_add_select(move, vehicle_type=3, stage='ground')
            return
        elif self.ground_formation_stage == 3 and self.lock == 'ground':  # add Tanks to selection
            self.act_add_select(move, vehicle_type=4, stage='ground')
            return
        elif self.ground_formation_stage == 4 and self.lock == 'ground':  # rotate all ground units
            self.act_rotate(move, types=[0, 3, 4], pivot=(122, 122), stage='ground')
            self.form_rotation_start = world.tick_index
            self.lock = 'free'
            return
        ## MOVE IN
        elif self.ground_formation_stage == 5:                            # wait for 300 ticks
            if world.tick_index - self.form_rotation_start >= 300:
                self.next_formation_stage(stage='ground')
        elif self.ground_formation_stage in [6, 7, 8, 9, 10, 11]:         # move each type of vehicle to the center
            ground_centroid = self.get_centroid(self.me, types=[0, 3, 4])
            type = [0, 3, 4][(self.ground_formation_stage-6)//2]
            type_centroid = self.get_centroid(self.me, types=[type])
            if self.get_squared_distance(*ground_centroid, *type_centroid) < self.EPS:
                pass
            if self.ground_formation_stage % 2 == 0 and self.lock == 'free':
                self.act_new_select(move, vehicle_type=type, stage='ground')
                self.lock = 'ground'
            elif self.lock == 'ground':
                self.act_move(move, source=type_centroid,
                                    dest=(122, 122), stage='ground')
                self.lock = 'free'
            return
        elif self.ground_formation_stage == 12:                           # wait for another 300 ticks
            if world.tick_index - self.form_rotation_start >= 600:
                self.next_formation_stage(stage='ground')
        elif self.ground_formation_stage == 13 and self.lock == 'free':   # select ARRVs
            self.act_new_select(move, vehicle_type=0, stage='ground')
            self.lock = 'ground'
            return
        elif self.ground_formation_stage == 14 and self.lock == 'ground': # add IFVs to selection
            self.act_add_select(move, vehicle_type=3, stage='ground')
            return
        elif self.ground_formation_stage == 15 and self.lock == 'ground': # add Tanks to selection
            self.act_add_select(move, vehicle_type=4, stage='ground')
            return
        elif self.ground_formation_stage == 16 and self.lock == 'ground': # scale down the formation of ground units
            self.act_scale(move, self.get_centroid(self.me, types=[0, 3, 4]),
                                                   factor=0.1, stage='ground')
            self.lock = 'free'
            return

        # DETECT ENEMY NUKES
        # NOTE: only does this after the formation
        if self.enemy_is_nuking == False and self.enemy.next_nuclear_strike_tick_index > 0:
            self.enemy_is_nuking = True
            self.en_nuke_detected = world.tick_index
            self.en_nuke_x = self.enemy.next_nuclear_strike_x
            self.en_nuke_y = self.enemy.next_nuclear_strike_y
            self.en_nuke_tick = self.enemy.next_nuclear_strike_tick_index
        elif self.enemy_is_nuking == True and self.enemy.next_nuclear_strike_tick_index < 0:
            self.enemy_is_nuking = False
            self.en_nuke_detected = -1
        # EVADE ENEMY NUKES
        if self.enemy_is_nuking == True:
            # OPTION: delay scale up when enemy is nuking to prevent the
            #         units from deviating too much from the formation
            if world.tick_index == self.en_nuke_detected:
                self.act_new_select(move)
            elif world.tick_index == self.en_nuke_detected + 1:
                self.act_scale(move, pivot=(self.en_nuke_x,
                                            self.en_nuke_y), factor=10.0)
            # do nothing while enemy is nuking
            return
        # REGROUP
        if self.enemy_is_nuking == False and world.tick_index <= self.en_nuke_tick + 32:
            if world.tick_index == self.en_nuke_tick + 1:
                self.act_new_select(move)
            elif world.tick_index == self.en_nuke_tick + 2:
                self.act_scale(move, pivot=(self.en_nuke_x,
                                            self.en_nuke_y), factor=0.1)
            # do nothing else while regrouping
            return

        # PHASE II: TRAVEL ACROSS GAME WORLD
        # OPTION: use potential fields to calculate movement patterns
        for idx in range(1000, 20001, 250):
            if world.tick_index == idx + 1:
                self.act_new_select(move)
            elif world.tick_index == idx + 2:
                me_x, me_y = median_me_all
                en_x, en_y = self.world.width, self.world.height
                for centroid in self.get_centroids(self.enemy, method='kmeans'):
                    if self.get_squared_distance(me_x, me_y, *centroid) < \
                       self.get_squared_distance(me_x, me_y, en_x, en_y):
                        en_x, en_y = centroid
                self.act_move(move, source=(me_x, me_y), dest=(en_x, en_y),
                              max_speed=game.tank_speed * game.swamp_terrain_speed_factor)
        # REFORM
        # Overrides previous loop
        for idx in range(1500, 20001, 1500):
            if world.tick_index == idx + 1:
                self.act_new_select(move, vehicle_type=1)
            elif world.tick_index == idx + 2:
                self.act_add_select(move, vehicle_type=2)
            elif world.tick_index == idx + 3:
                self.act_move(move, source=median_me_aerial, dest=median_me_ground)
            elif world.tick_index == idx + 4:
                self.act_new_select(move, vehicle_type=0)
            elif world.tick_index == idx + 5:
                self.act_add_select(move, vehicle_type=3)
            elif world.tick_index == idx + 6:
                self.act_add_select(move, vehicle_type=4)
            elif world.tick_index == idx + 7:
                self.act_rotate(move, types=[0, 3, 4], angle=np.pi/2)
            if world.tick_index == 250 + idx + 1:
                self.act_new_select(move)
            elif world.tick_index == 250 + idx + 2:
                self.act_scale(move, pivot=median_me_all, factor=0.1)

        # NUCLEAR OPTION (not really an option; all hail america!)
        # Only done when the strategy is idle
        if world.tick_index % 10 == 0 and self.can_nuke() and move.action is None:
            best_nuke_score = 0
            # TODO: Take the velocity of the target into account
            # TODO: Take your velocity into account for planning when to nuke
            for centroid in self.get_centroids(self.enemy, method='kmeans'):
                unit_nuking = self.get_nuking_unit(*centroid)
                damages = self.calc_nuke_damages(*centroid)
                if unit_nuking == -1 or damages[1] < self.EPS:
                    continue
                if damages[0]/damages[1] <= self.SIG_NUKE_DAMAGE_RATIO \
                    and damages[1] - damages[0] > best_nuke_score:
                        move.action = ActionType.TACTICAL_NUCLEAR_STRIKE
                        move.x, move.y = centroid
                        move.vehicle_id = unit_nuking
                        break
            if move.action == ActionType.TACTICAL_NUCLEAR_STRIKE:
                print("NUKE: ({}, {}) with unit #{}".format(move.x, move.y,
                                                            move.vehicle_id))


    def update(self, me, world, game):
        self.me = me
        self.enemy = world.get_opponent_player()
        self.world = world
        self.game = game
        for unit in world.new_vehicles:
            self.all_vehicles[unit.id] = unit
        for unit_update in world.vehicle_updates:
            unit = self.all_vehicles[unit_update.id]
            unit.x = unit_update.x
            unit.y = unit_update.y
            unit.durability = unit_update.durability
            unit.remaining_attack_cooldown_ticks = unit_update.remaining_attack_cooldown_ticks
            unit.selected = unit_update.selected
            unit.groups = unit_update.groups

    def clear_cache(self):
        pass

    def next_formation_stage(self, stage=None):
        if stage == 'ground':
            self.ground_formation_stage += 1
        elif stage == 'aerial':
            self.aerial_formation_stage += 1


    def get_units(self, player, types=all_vehicle_types, only_selected=False):
        for id in self.all_vehicles:
            unit = self.all_vehicles[id]
            if unit.player_id == player.id and unit.type in types:
                if (only_selected and not unit.selected) or unit.durability < self.EPS:
                    continue
                yield unit

    def get_data_of_units(self, player, types=all_vehicle_types, only_selected=False):
        min_x, max_x = self.world.width, 0
        min_y, max_y = self.world.height, 0
        positions = []
        for unit in self.get_units(player, types, only_selected):
            min_x, max_x = min(min_x, unit.x), max(max_x, unit.x)
            min_y, max_y = min(min_y, unit.y), max(max_y, unit.y)
            positions.append([unit.x, unit.y])
        return (min_x, max_x), (min_y, max_y), np.array(positions)

    def get_cell_properties_of_unit(self, unit):
        cell_x = int(unit.x/32.0)
        cell_y = int(unit.y/32.0)
        weather = self.world.weather_by_cell_x_y[cell_x][cell_y]
        terrain = self.world.terrain_by_cell_x_y[cell_x][cell_y]
        return weather, terrain

    def get_vision_range_factor_of_unit(self, unit):
        weather, terrain = self.get_cell_properties_of_unit(unit)
        if unit.aerial:
            return [1.0, 0.8, 0.6][weather]
        else:
            return [1.0, 1.0, 0.8][terrain]

    def get_speed_factor_of_unit(self, unit):
        weather, terrain = self.get_cell_properties_of_unit(unit)
        if unit.aerial:
            return [1.0, 0.8, 0.6][weather]
        else:
            return [1.0, 0.6, 0.8][terrain]

    def get_squared_distance(self, x1, y1, x2, y2):
        return (x1 - x2)**2 + (y1 - y2)**2

    def get_distance(self, x1, y1, x2, y2):
        return np.sqrt(get_squared_distance(x1, y1, x2, y2))

    def get_median(self, player, types=all_vehicle_types, only_selected=False):
        positions = self.get_data_of_units(player, types, only_selected)[2]
        if len(positions) > 0:
            return np.median(positions, axis=0)
        else:
            return self.world.width, self.world.height

    def get_centroid(self, player, types=all_vehicle_types, only_selected=False):
        positions = self.get_data_of_units(player, types, only_selected)[2]
        if len(positions) > 0:
            return np.mean(positions, axis=0)
        else:
            return self.world.width, self.world.height

    def get_centroids(self, player, types=all_vehicle_types, only_selected=False,
                      method='kmeans', num_clusters=5, iter=5, minit='points',
                      do_elbow_analysis=True, linkage_method='single',
                      clustering_method='maxclust', max_dist=100.0):
        positions = self.get_data_of_units(player, types, only_selected)[2]
        if len(positions) > 0:
            # print("GET CENTROIDS", method)
            if method == 'kmeans':
                centroids, clustering = kmeans(positions,
                                               min(num_clusters, len(positions)),
                                               iter,
                                               minit)
                for centroid in centroids:
                    yield centroid
            elif method == 'hierarchal':
                dend = hac.linkage(positions, method=linkage_method)
                if clustering_method == 'maxclust':
                    if do_elbow_analysis:
                        knee = np.diff(dend[::-1, 2], 2)
                        num_clusters = knee.argmax() + 2
                    clustering = hac.fcluster(dend, num_clusters, 'maxclust')
                elif clustering_method == 'distance':
                    clustering = hac.fcluster(dend, max_dist, 'distance')

                for cluster in set(clustering):
                    yield np.mean(positions[clustering == cluster], axis=0)


    def get_nuking_unit(self, x, y):
        # OPTION: get the farthest of all the possible units to prevent it
        #         from getting killed
        # OPTION: or just get the optimal unit for the job (taking everything
        #         into account)
        for unit_id in self.all_vehicles:
            unit = self.all_vehicles[unit_id]
            if unit.durability < self.EPS or unit.player_id == self.enemy.id:
                continue
            mod_vision_range = unit.vision_range * self.get_vision_range_factor_of_unit(unit)
            if unit.get_squared_distance_to(x, y) <= mod_vision_range**2:
                return unit_id
        return -1

    def can_nuke(self):
        return self.me.remaining_nuclear_strike_cooldown_ticks <= 0

    def calc_nuke_damages(self, x, y):
        # TODO: take durability and damage into account
        # OPTION: disregard some units like the healers of the enemy
        me_dam = 0
        en_dam = 0
        for unit in self.get_units(self.me):
            dist = unit.get_distance_to(x, y)
            if dist <= 50:
                me_dam += 90 * (50 - dist)
        for unit in self.get_units(self.enemy):
            dist = unit.get_distance_to(x, y)
            if dist <= 50:
                en_dam += 90 * (50 - dist)
        return me_dam, en_dam


    def act_new_select(self, m, vehicle_type=None, domain=None, range=None, stage=None):
        m.action = ActionType.CLEAR_AND_SELECT
        m.vehicle_type = vehicle_type
        m.left, m.right = (0.0, self.world.width) if domain is None else domain
        m.top, m.bottom = (0.0, self.world.height) if range is None else range
        self.next_formation_stage(stage)
        print(self.aerial_formation_stage, self.ground_formation_stage,
              "CLEAR AND SELECT", m.vehicle_type)

    def act_move(self, m, relative=None, source=None, dest=None, max_speed=0.0, stage=None):
        m.action = ActionType.MOVE
        m.max_speed = max_speed
        if hasattr(relative, "__len__"):
            m.x, m.y = relative
        elif hasattr(source, "__len__") and hasattr(dest, "__len__"):
            m.x = dest[0] - source[0]
            m.y = dest[1] - source[1]
        self.next_formation_stage(stage)
        print(self.aerial_formation_stage, self.ground_formation_stage,
              "MOVE", m.x, m.y)

    def act_scale(self, m, pivot, factor=None, stage=None):
        m.action = ActionType.SCALE
        m.x, m.y = pivot
        m.factor = self.AERIAL_SCALE_FACTOR if factor is None else factor
        self.next_formation_stage(stage)
        print(self.aerial_formation_stage, self.ground_formation_stage,
              "SCALE", m.x, m.y, m.factor)

    def act_add_select(self, m, vehicle_type, stage=None):
        m.action = ActionType.ADD_TO_SELECTION
        m.right = self.world.width
        m.bottom = self.world.height
        m.vehicle_type = vehicle_type
        self.next_formation_stage(stage)
        print(self.aerial_formation_stage, self.ground_formation_stage,
              "ADD TO SELECTION", m.vehicle_type)

    def act_rotate(self, m, types, pivot='auto', angle=np.pi, stage=None):
        m.action = ActionType.ROTATE
        m.angle = angle
        if pivot == 'auto':
            m.x, m.y = self.get_median(self.me, types)
        else:
            m.x, m.y = pivot
        self.next_formation_stage(stage)
        print(self.aerial_formation_stage, self.ground_formation_stage,
              "ROTATE", m.x, m.y, m.angle)

    def check_in_pos(self, current, target):
        return abs(current[0]-target[0]) < self.EPS and \
               abs(current[1]-target[1]) < self.EPS

    def check_if_scaled(self, vehicle_type, target_size):
        x_range, y_range = self.get_data_of_units(self.me, [vehicle_type])[0:2]
        return ((x_range[1] - x_range[0]) - target_size) < self.EPS and \
               ((y_range[1] - y_range[0]) - target_size) < self.EPS
