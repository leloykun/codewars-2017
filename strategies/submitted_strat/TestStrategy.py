from model.ActionType import ActionType
from model.VehicleType import VehicleType
from model.Game import Game
from model.Move import Move
from model.Player import Player
from model.World import World

import copy
import random
import math
from collections import deque, namedtuple

import numpy as np
from scipy.cluster.vq import kmeans2 as kmeans
import scipy.cluster.hierarchy as hac

# Use potential fields in occupation of facilities / attacking enemy
# separate new units into their own groups


class Bfs:
    dr = [0, 0, 0, -1, 1]
    dc = [0, -1, 1, 0, 0]

    def in_corner(self, p):
        if p[0] == 1 or p[1] == 1:
            return False
        return True

    def check_done(self, ar):
        for unit in range(3):
            if not self.in_corner(ar[unit]):
                return False
        return True

    def check_valid(self, ar):
        for unit in range(3):
            for i in range(2):
                if not (0 <= ar[unit][i] <= 2):
                    return False
        if len(set(str(x) for x in ar)) != 3:
            return False
        return True

    def solve(self, ar):
        q = deque()
        q.append((ar, [ar]))

        vis = set()

        while len(q) > 0:
            ar, hist = q.popleft()
            vis.add(str(ar))
            ars = set(str(x) for x in ar)
            if self.check_done(ar):
                return hist

            for k1 in range(5):
                for k2 in range(5):
                    for k3 in range(5):
                        tar = copy.deepcopy(ar)
                        temp_hist = copy.deepcopy(hist)
                        #if not self.in_corner(tar[0]):
                        tar[0][0] += self.dr[k1]
                        tar[0][1] += self.dc[k1]
                        if str(tar[0]) in ars and k1 != 0:
                            continue
                        #if not self.in_corner(tar[1]):
                        tar[1][0] += self.dr[k2]
                        tar[1][1] += self.dc[k2]
                        if str(tar[1]) in ars and k2 != 0:
                            continue
                        #if not self.in_corner(tar[2]):
                        tar[2][0] += self.dr[k3]
                        tar[2][1] += self.dc[k3]
                        if str(tar[2]) in ars and k3 != 0:
                            continue
                        if self.check_valid(tar) and not (str(tar) in vis):
                            temp_hist.append(tar)
                            q.append((tar, temp_hist))


class Group:
    VEHICLE = [1, 2, 3, 4, 5]
    ARRV = 1
    FIGHTER = 2
    HELICOPTER = 3
    IFV = 4
    TANK = 5

    AERIAL = 6
    GROUND = 7

    ALL = 8


class MyStrategy:
    WORLD_WIDTH, WORLD_HEIGHT = 1024, 1024
    EPS = 0.1

    ALL_VEHICLE_TYPES = [0, 1, 2, 3, 4]
    VEHICLE_TYPE_NAME = ['ARRV', 'FIGHTER', 'HELICOPTER', 'IFV', 'TANK']

    BASE_SQUAD_SIZE = 54
    AERIAL_SCALE_FACTOR = 1.4
    GROUND_SCALE_FACTOR = 2.1
    AERIAL_SCALED_SQUAD_SIZE = BASE_SQUAD_SIZE * AERIAL_SCALE_FACTOR
    GROUND_SCALED_SQUAD_SIZE = BASE_SQUAD_SIZE * GROUND_SCALE_FACTOR

    all_vehicles = {}
    m_queue = deque()

    aerial_form_stage = 1
    ground_form_stage = 1
    aerial_form_done = False
    ground_form_done = False
    shrinking_done = False

    ground_form_time_stamp = -1

    en_is_nuking = False
    en_nuke_tick_detected = -1
    en_nuke_tick_drop = -1
    en_nuke_x, en_nuke_y = -1, -1

    init_grid = [[-1, -1, -1],
                 [-1, -1, -1],
                 [-1, -1, -1]]
    ground_grev = [[-1, -1],
                   [-1, -1],
                   [-1, -1]]
    aerial_grev = [[-1, -1],
                   [-1, -1]]
    init_min = [[-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1]]
    init_max = [[-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1]]
    init_p2i = {45:0,
                119:1,
                193:2}

    bfs = Bfs()

    facilities_owned_now = {}
    facilities_owned_prev = {}

    unit_prod = 0


    def init(self):
        self.world_has_facilities = (len(self.world.facilities) > 0)

        for type in self.ALL_VEHICLE_TYPES:
            med = self.get_median(player=self.me, types=[type])
            pos = self.get_positions(player=self.me, types=[type])
            self.init_min[type] = np.min(pos, axis=0)
            self.init_max[type] = np.max(pos, axis=0)
            print(self.VEHICLE_TYPE_NAME[type])
            print("  median:", med)
            print("  min:", self.init_min[type])
            print("  max:", self.init_max[type])
            i = self.init_p2i[int(med[0])]
            j = self.init_p2i[int(med[1])]
            self.init_grid[j][i] = type
            if type == 0:
                self.ground_grev[0] = [i, j]
            elif type == 1:
                self.aerial_grev[0] = [i, j]
            elif type == 2:
                self.aerial_grev[1] = [i, j]
            elif type == 3:
                self.ground_grev[1] = [i, j]
            elif type == 4:
                self.ground_grev[2] = [i, j]
        print("GRID:", self.init_grid)
        print()

        self.current_grev = self.ground_grev
        self.solution_grev = self.bfs.solve(self.current_grev)
        print(self.solution_grev)
        print()

        f_x, f_y = self.aerial_grev[0]
        h_x, h_y = self.aerial_grev[1]
        if (f_x == h_x and f_y < h_y) or (f_x > f_y and not
                                          (f_x < h_x and f_y == h_y)):
            self.aerial_form_dir = 'right_down'
        else:
            self.aerial_form_dir = 'down_right'

        # GROUP {1: ARRV,
        #        2: FIGHTER,
        #        3: HELICOPTER,
        #        4: IFV,
        #        5: TANK}
        for type in self.ALL_VEHICLE_TYPES:
            self.m_queue.append(self.new_move(action=1, vehicle_type=type))
            self.m_queue.append(self.new_move(action=4, group=Group.VEHICLE[type]))
        # GROUP 6: Aerial Units
        self.m_queue.append(self.new_move(action=1, vehicle_type=1))
        self.m_queue.append(self.new_move(action=2, vehicle_type=2))
        self.m_queue.append(self.new_move(action=4, group=Group.AERIAL))
        # GROUP 7: Ground Units
        self.m_queue.append(self.new_move(action=1))
        self.m_queue.append(self.new_move(action=3, group=Group.AERIAL))
        self.m_queue.append(self.new_move(action=4, group=Group.GROUND))
        # GROUP 8: All Units
        self.m_queue.append(self.new_move(action=1))
        self.m_queue.append(self.new_move(action=4, group=Group.ALL))

    def update(self, me: Player, en: Player, world: World, game: Game):
        self.me = me
        self.en = en
        self.world = world
        self.game = game
        # init new vehicles
        for unit in self.world.new_vehicles:
            unit.update_tick = self.world.tick_index
            self.all_vehicles[unit.id] = unit
        # update existing vehicles
        for unit_update in self.world.vehicle_updates:
            if unit_update.durability == 0:
                self.all_vehicles.pop(unit_update.id)
            else:
                unit = self.all_vehicles[unit_update.id]
                if (unit_update.x != unit.x or unit_update.y != unit.y):
                    unit.update_tick = self.world.tick_index
                unit.update(unit_update)
        # update facilities
        self.facilities_owned_now = {}
        for facility in self.world.facilities:
            if facility.owner_player_id == self.me.id:
                self.facilities_owned_now[facility.id] = facility
                if (facility.vehicle_type is None and facility.type == 1 and
                        self.aerial_form_done and self.ground_form_done and
                        self.count_units_on_facility([self.me], facility) == 0):
                    self.m_queue.append(self.new_move(action=10, 
                                                      facility_id=facility.id,
                                                      vehicle_type=2))
                    self.unit_prod = (self.unit_prod + 1) % 5

    def move(self, me: Player, world: World, game: Game, move: Move):
        en = world.get_opponent_player()
        self.update(me, en, world, game)

        if world.tick_index == 0:
            self.init()

        if me.remaining_action_cooldown_ticks > 0:
            return

        if self.execute_move_queue(move):
            return

        # formation
        if not self.aerial_form_done:
            self.do_aerial_formation()
        if not self.ground_form_done:
            self.do_ground_formation_2()
        if self.execute_move_queue(move):
            return

        # shrink
        if self.ground_form_done and self.aerial_form_done and not self.shrinking_done:
            self.do_shrinking()
        if self.execute_move_queue(move):
            return

        # evade nukes
        if self.aerial_form_done and self.ground_form_done:
            self.evade_nukes()
        if self.execute_move_queue(move):
            return

        #self.fix_formation()
        #if self.execute_move_queue(move):
        #    return

        # attack signature
        if (self.units_have_stopped(self.me, group=Group.AERIAL) and 
                self.units_have_stopped(self.me, group=Group.GROUND)):
            if self.world_has_facilities:
                self.occupy_nearest_facility()
            else:
                self.attack_nearest_cluster_of_enemy()
        if self.execute_move_queue(move):
            return

        if self.can_nuke() and self.world.tick_index % 10 == 0:
            self.fire_nukes()
        self.execute_move_queue(move)


    def execute_move_queue(self, m):
        if len(self.m_queue) == 0:
            return False
        temp_m = self.m_queue.popleft()
        for key in vars(temp_m).keys():
            value = getattr(temp_m, key)
            setattr(m, key, value)
        return True

    def do_aerial_formation(self):
        f_pos = self.get_positions(self.me, group=Group.FIGHTER)
        (f_min_x, f_min_y), (f_max_x, f_max_y) = self.get_min_max(f_pos)
        f_med_x, f_med_y = self.get_median(f_pos)

        h_pos = self.get_positions(self.me, group=Group.HELICOPTER)
        (h_min_x, h_min_y), (h_max_x, h_max_y) = self.get_min_max(h_pos)
        h_med_x, h_med_y = self.get_median(h_pos)

        g_pos = self.get_positions(self.me, group=Group.GROUND)
        (g_min_x, g_min_y), (g_max_x, g_max_y) = self.get_min_max(g_pos)
        g_med_x, g_med_y = self.get_median(g_pos)

        if self.aerial_form_stage == 1:                                         # move fighter planes away from initial grid
            if self.aerial_form_dir == 'right_down':
                dx, dy = 270 - f_med_x, 0
            else:
                dx, dy = 0, 270 - f_med_y
            self.m_queue.append(self.new_move(action=1, group=Group.FIGHTER))
            self.m_queue.append(self.new_move(action=7, vector=(dx, dy)))
            self.aerial_form_stage += 1
        elif (self.aerial_form_stage == 2 and                                   # ready aerial units for sifting
              self.units_have_stopped(self.me, group=Group.AERIAL, duration=5)):
            # move fighter planes first
            if self.aerial_form_dir == 'right_down':
                dx, dy = 0, 270 - f_med_y
            else:
                dx, dy = 270 - f_med_x, 0
            self.m_queue.append(self.new_move(action=1, group=Group.FIGHTER))
            self.m_queue.append(self.new_move(action=7, vector=(dx, dy)))
            # then helicopters
            if self.aerial_form_dir == 'right_down':
                dx = 274.2 - self.AERIAL_SCALED_SQUAD_SIZE - 10 - h_med_x
                dy = 274.2 - h_med_y
            else:
                dx = 274.2 - h_med_x
                dy = 274.2 - self.AERIAL_SCALED_SQUAD_SIZE - 10 - h_med_y
            self.m_queue.append(self.new_move(action=1, group=Group.HELICOPTER))
            self.m_queue.append(self.new_move(action=7, vector=(dx, dy)))
            self.aerial_form_stage += 1
        elif (self.aerial_form_stage == 3 and                                   # scale both types of aerial vehicles
              self.units_have_stopped(self.me, group=Group.AERIAL, duration=5)):
            self.m_queue.append(self.new_move(action=1, group=Group.FIGHTER))
            self.m_queue.append(self.new_move(action=9,
                                              factor=1.4,
                                              vector=(f_med_x, f_med_y)))
            self.m_queue.append(self.new_move(action=1, group=Group.HELICOPTER))
            self.m_queue.append(self.new_move(action=9,
                                              factor=1.4,
                                              vector=(h_med_x, h_med_y)))
            self.aerial_form_stage += 1
        elif (self.aerial_form_stage == 4 and                                   # sift
              self.units_have_stopped(self.me, group=Group.AERIAL, duration=5)):
            dx = 274.2 - h_med_x
            dy = 274.2 - h_med_y
            self.m_queue.append(self.new_move(action=1, group=Group.HELICOPTER))
            self.m_queue.append(self.new_move(action=7, vector=(dx, dy)))
            self.aerial_form_stage += 1
        elif (self.aerial_form_stage == 5 and                                   # move to centroid of ground units
              self.units_have_stopped(self.me, group=Group.AERIAL, duration=5)):
            dx = 119 - h_med_x
            dy = 119 - h_med_y
            self.m_queue.append(self.new_move(action=1, group=Group.AERIAL))
            self.m_queue.append(self.new_move(action=7, vector=(dx, dy)))
            self.aerial_form_stage += 1
        elif (self.aerial_form_stage >= 6 and                                   # done!
              self.units_have_stopped(self.me, group=Group.AERIAL, duration=5)):
            self.aerial_form_done = True

    def do_ground_formation_1(self):
        ### DEPRECATED ##
        g_pos = self.get_positions(self.me, types=[0, 3, 4])
        (g_min_x, g_min_y), (g_max_x, g_max_y) = self.get_min_max(g_pos)
        g_med_x, g_med_y = self.get_median(g_pos)
        g_cen_x, g_cen_y  = self.get_centroid(g_pos)

        if self.ground_form_stage == 1:                                         # rotate
            self.m_queue.append(self.new_move(action=1, group=Group.GROUND))
            self.m_queue.append(self.new_move(action=8, vector=(119, 119),
                                              angle=np.pi/2))
            self.ground_rotate_start = self.world.tick_index
            self.ground_form_stage += 1
        elif (self.ground_form_stage == 2 and                                   # pack-in
              self.world.tick_index >= self.ground_rotate_start + 300):
            types = [0, 3, 4]
            for type in types:
                type_centroid = self.get_centroid(player=self.me, types=[type])
                self.m_queue.append(self.new_move(action=1, vehicle_type=type))
                dx = g_cen_x - type_centroid[0]
                dy = g_cen_y - type_centroid[1]
                self.m_queue.append(self.new_move(action=7, vector=(dx, dy)))
            self.ground_form_stage += 1
        elif (self.ground_form_stage == 3 and                                   # scale down
              self.world.tick_index >= self.ground_rotate_start + 600):
            self.m_queue.append(self.new_move(action=1, group=Group.GROUND))
            self.m_queue.append(self.new_move(action=9,
                                              factor=0.1,
                                              vector=(g_cen_x, g_cen_y)))
            self.ground_form_stage += 1
        elif self.ground_form_stage >= 4:                                       # done!
            self.ground_form_done = True

    def do_ground_formation_2(self):
        g_types = [0, 3, 4]
        g_idx = [2, 1, 0]
        if self.ground_form_stage == 1:                                         # move to position p1
            if len(self.solution_grev) <= 1:
                self.ground_form_stage += 1
                return
            for i in g_idx:
                type = g_types[i]
                dx = 74 * (self.solution_grev[1][i][0] -
                           self.solution_grev[0][i][0])
                dy = 74 * (self.solution_grev[1][i][1] -
                           self.solution_grev[0][i][1])
                self.m_queue.append(self.new_move(action=1, group=Group.VEHICLE[type]))
                self.m_queue.append(self.new_move(action=7, vector=(dx, dy)))
            self.ground_form_stage += 1
        elif (self.ground_form_stage == 2 and                                   # move to position p2
                self.units_have_stopped(self.me, group=Group.GROUND, duration=5)):
            if len(self.solution_grev) <= 2:
                self.ground_form_stage += 1
                return
            for i in g_idx:
                type = g_types[i]
                dx = 74 * (self.solution_grev[2][i][0] -
                           self.solution_grev[1][i][0])
                dy = 74 * (self.solution_grev[2][i][1] -
                           self.solution_grev[1][i][1])
                self.m_queue.append(self.new_move(action=1, group=Group.VEHICLE[type]))
                self.m_queue.append(self.new_move(action=7, vector=(dx, dy)))
            self.ground_form_stage += 1
        elif (self.ground_form_stage == 3 and                                   # prepare for sifting
              self.units_have_stopped(self.me, group=Group.GROUND, duration=5)):
            for k in g_idx:
                type = g_types[k]
                med_p = self.get_median(player=self.me, types=[type])
                dx = 0
                dy = 4.2 if med_p[0] > 100 else 0
                self.m_queue.append(self.new_move(action=1, group=Group.VEHICLE[type]))
                self.m_queue.append(self.new_move(action=7, vector=(dx, dy)))
            self.ground_form_stage += 1
        elif (self.ground_form_stage == 4 and                                   # scale up
              self.units_have_stopped(self.me, group=Group.GROUND, duration=5)):
            for k in g_idx:
                type = g_types[k]
                med_p = self.get_median(player=self.me, types=[type])
                min_p, max_p = self.get_min_max(self.get_positions(self.me, types=[type]))
                x = min_p[0] if med_p[0] < 100 else max_p[0]
                y = min_p[1] if med_p[1] < 100 else max_p[1]
                self.m_queue.append(self.new_move(action=1, group=Group.VEHICLE[type]))
                self.m_queue.append(self.new_move(action=9,
                                                  factor=1.4,
                                                  vector=(x, y)))
            self.ground_form_stage += 1
        elif (self.ground_form_stage == 5 and                                   # move to center (on x-axis)
              self.units_have_stopped(self.me, group=Group.GROUND, duration=5)):
            for k in g_idx:
                type = g_types[k]
                med_p = self.get_median(player=self.me, types=[type])
                dx = 119 - med_p[0] - (4.2 if med_p[1] > 100 else 0)
                dy = 0
                self.m_queue.append(self.new_move(action=1, group=Group.VEHICLE[type]))
                self.m_queue.append(self.new_move(action=7, vector=(dx, dy)))
            self.ground_form_time_stamp = self.world.tick_index
            self.ground_form_stage += 1
        elif (self.ground_form_stage == 6 and                                   # move to center (on y-axis)
              self.units_have_stopped(self.me, group=Group.GROUND, duration=5)):
            for k in g_idx:
                type = g_types[k]
                med_p = self.get_median(player=self.me, types=[type])
                dx = 0
                dy = 119 - med_p[1]
                self.m_queue.append(self.new_move(action=1, group=Group.VEHICLE[type]))
                self.m_queue.append(self.new_move(action=7, vector=(dx, dy)))
            self.ground_form_stage += 1
        elif (self.ground_form_stage >= 7 and                                   # done!
              self.units_have_stopped(self.me, group=Group.GROUND, duration=5)):
            self.ground_form_done = True

    def do_shrinking(self):
        self.m_queue.append(self.new_move(action=1, group=Group.ALL))
        self.m_queue.append(self.new_move(action=9,
                                          factor=0.1,
                                          vector=self.get_centroid(player=self.me)))
        self.shrinking_done = True

    def fix_formation(self):
        a_pos = self.get_positions(self.me, types=[1, 2])
        a_cen_x, a_cen_y = self.get_centroid(a_pos)

        g_pos = self.get_positions(self.me, types=[0, 3, 4])
        g_cen_x, g_cen_y = self.get_centroid(g_pos)

        '''if (self.world_has_facilities and 
            self.world.tick_index >= 1500 and
            self.world.tick_index % 1500 == 0):
                # move aerial units to center of ground units
                self.m_queue.append(self.new_move(action=1, group=group.AERIAL))
                dx = g_cen_x - a_cen_x
                dy = g_cen_y - a_cen_y
                self.m_queue.append(self.new_move(action=7, vector=(dx, dy)))
                # rotate ground units
                self.m_queue.append(self.new_move(action=1, group=group.GROUND))
                self.m_queue.append(self.new_move(action=8,
                                                  angle=np.pi/2,
                                                  vector=(g_cen_x, g_cen_y)))'''
        if (self.world.tick_index >= 1750 and
              (self.world.tick_index-250) % 1500 == 0):
            for group in [1, 2]:
                centroid = self.get_centroid(player=self.me, group=group)
                self.m_queue.append(self.new_move(action=1, group=group))
                self.m_queue.append(self.new_move(action=9,
                                                  factor=0.1,
                                                  vector=centroid))


    def get_nearest_cluster_of_enemy(self, x, y):
        en_x, en_y = self.world.width, self.world.height
        for centroid in self.get_centroids(self.en, method='kmeans'):
            if self.get_squared_dist(x, y, *centroid) < \
               self.get_squared_dist(x, y, en_x, en_y):
                en_x, en_y = centroid
        return en_x, en_y

    def attack_nearest_cluster_of_enemy(self):
        me_x, me_y = self.get_centroid(player=self.me, group=Group.ALL)
        en_x, en_y = self.get_nearest_cluster_of_enemy(me_x, me_y)
        dx = en_x - me_x
        dy = en_y - me_y
        self.m_queue.append(self.new_move(action=1))
        self.m_queue.append(self.new_move(action=7, vector=(dx, dy),
                                          max_speed=0.18))

    def occupy_nearest_facility(self):
        for group in [Group.ALL]:
            me_x, me_y = self.get_centroid(player=self.me, group=group)

            min_dist = 1024**2
            x, y = self.world.width, self.world.height
            for facility in self.world.facilities:
                if facility.owner_player_id == self.me.id:
                    continue
                fac_x = facility.left + self.game.facility_width/2.0
                fac_y = facility.top + self.game.facility_height/2.0
                fac_x = min(max(50, fac_x), self.WORLD_WIDTH-50)
                fac_y = min(max(50, fac_y), self.WORLD_HEIGHT-50)
                dist = self.get_squared_dist(me_x, me_y, fac_x, fac_y)
                if dist < min_dist:
                    min_dist = dist
                    x, y = fac_x, fac_y
            # get dx, dy
            dx = x - me_x
            dy = y - me_y
            # normalize
            r = np.hypot(dx, dy)
            dx = dx * (16/r)
            dy = dy * (16/r)

            max_speed = self.get_slowest_unit_speed(self.me)

            self.m_queue.append(self.new_move(action=1, group=group))
            self.m_queue.append(self.new_move(action=7, vector=(dx, dy),
                                              max_speed=max_speed))


    def evade_nukes(self):
        if not self.en_is_nuking and self.en.next_nuclear_strike_tick_index > 0:
            self.en_is_nuking = True
            self.en_nuke_tick_detected = self.world.tick_index
            self.en_nuke_tick_drop = self.en.next_nuclear_strike_tick_index
            self.en_nuke_x = self.en.next_nuclear_strike_x
            self.en_nuke_y = self.en.next_nuclear_strike_y
        elif self.en_is_nuking and self.en.next_nuclear_strike_tick_index < 0:
            self.en_is_nuking = False
            self.en_nuke_tick_detected = -1
        if self.en_is_nuking:
            for group in [Group.ALL]:
                centroid = self.get_centroid(player=self.me, group=group)
                if self.get_dist(*centroid, self.en_nuke_x, self.en_nuke_y) > 100:
                    continue
                # NOTE: INVERSELY ORDERED
                self.m_queue.appendleft(self.new_move(action=7))
                self.m_queue.appendleft(self.new_move(action=1, group=group))
                for _ in range(30):
                    self.m_queue.appendleft(self.new_move())
                self.m_queue.appendleft(self.new_move(action=9,
                                                      factor=0.1,
                                                      vector=(self.en_nuke_x,
                                                              self.en_nuke_y)))
                self.m_queue.appendleft(self.new_move(action=1, group=group))
                for _ in range(30):
                    self.m_queue.appendleft(self.new_move())
                self.m_queue.appendleft(self.new_move(action=9,
                                                      factor=10.0,
                                                      vector=(self.en_nuke_x,
                                                              self.en_nuke_y)))
                self.m_queue.appendleft(self.new_move(action=1, group=group))

    def fire_nukes(self):
        max_damage = 0
        for centroid in self.get_centroids(self.en, method='kmeans'):
            nuking_unit = self.get_nuking_unit(*centroid)
            damages = self.calc_nuke_damages(*centroid)
            if nuking_unit == -1 or damages[1] < self.EPS:
                continue
            if damages[1] - damages[0] > max_damage:
                max_damage = damages[1] - damages[0]
                self.m_queue.append(self.new_move(action=11,
                                                  vector=centroid,
                                                  vehicle_id=nuking_unit))
        if len(self.m_queue) > 0:
            print("NUKE:", self.m_queue[-1].x, self.m_queue[-1].y)

    def get_nuking_unit(self, x, y):
        for id in self.all_vehicles:
            unit = self.all_vehicles[id]
            if unit.player_id == self.en.id:
                continue
            mod_vision_range = unit.vision_range * self.get_vision_range_factor(unit)
            if unit.get_squared_distance_to(x, y) <= 0.9*mod_vision_range**2:
                return id
        return -1

    def can_nuke(self):
        return self.me.remaining_nuclear_strike_cooldown_ticks <= 0

    def calc_nuke_damage_to_units(self, units, x, y):
        sum_dam = 0
        for unit in units:
            dist = unit.get_distance_to(x, y)
            if dist > 50:
                continue
            dam = 90 * (50 - dist)
            if dam > unit.durability:
                dam *= 2
            sum_dam += dam
        return sum_dam

    def calc_nuke_damages(self, x, y):
        me_dam = self.calc_nuke_damage_to_units(self.get_units(self.me), x, y)
        en_dam = self.calc_nuke_damage_to_units(self.get_units(self.en), x, y)
        return me_dam, en_dam


    def get_cell_properties_of_unit(self, unit):
        cell_x = int(unit.x/32.0)
        cell_y = int(unit.y/32.0)
        weather = self.world.weather_by_cell_x_y[cell_x][cell_y]
        terrain = self.world.terrain_by_cell_x_y[cell_x][cell_y]
        return weather, terrain

    def get_vision_range_factor(self, unit):
        weather, terrain = self.get_cell_properties_of_unit(unit)
        if unit.aerial:
            return [1.0, 0.8, 0.6][weather]
        else:
            return [1.0, 1.0, 0.8][terrain]

    def get_speed_factor(self, unit):
        weather, terrain = self.get_cell_properties_of_unit(unit)
        if unit.aerial:
            return [1.0, 0.8, 0.6][weather]
        else:
            return [1.0, 0.6, 0.8][terrain]

    def get_speed(self, unit=None, type=None):
        if type is None:
            type = unit.type
        return [0.4, 1.2, 0.9, 0.4, 0.3][type]


    def  get_slowest_unit_speed(self, player, types=ALL_VEHICLE_TYPES, group=0):
        speeds = []
        for unit in self.get_units(player, types, group):
            speeds.append(self.get_speed(unit) *
                          self.get_speed_factor(unit))
        return np.min(speeds)

    def count_units_on_facility(self, players, facility):
        x0, x1 = facility.left, facility.left + self.game.facility_width
        y0, y1 = facility.top, facility.top + self.game.facility_height
        cnt = 0
        for player in players:
            for unit in self.get_units(player):
                if x0 <= unit.x <= x1 and y0 <= unit.y <= y1:
                    cnt += 1
        return cnt

    def units_have_stopped(self, player, types=ALL_VEHICLE_TYPES, group=0,
                           duration=5):
        for unit in self.get_units(player, types, group):
            if self.world.tick_index - unit.update_tick < duration:
                return False
        return True

    def get_units(self, player, types=ALL_VEHICLE_TYPES, group=0):
        for id in self.all_vehicles:
            unit = self.all_vehicles[id]
            if unit.player_id == player.id and unit.type in types and \
               (group == 0 or (group != 0 and group in unit.groups)):
                yield unit

    def get_positions(self, player=None, types=ALL_VEHICLE_TYPES, group=0):
        positions = []
        for unit in self.get_units(player, types, group):
            positions.append([unit.x, unit.y])
        return np.array(positions)

    def get_min_max(self, positions):
        if len(positions) > 0:
            return np.min(positions, axis=0), np.max(positions, axis=0)
        else:
            return np.array([0, 0]), np.array([1024, 1024])

    def get_median(self, positions=None, player=None, types=ALL_VEHICLE_TYPES):
        if positions is None:
            positions = self.get_positions(player=player, types=types)
        if len(positions) > 0:
            return np.median(positions, axis=0)
        else:
            return self.world.width/2.0, self.world.height/2.0

    def get_centroid(self, positions=None, player=None, types=ALL_VEHICLE_TYPES,
                     group=0):
        if positions is None:
            positions = self.get_positions(player, types, group)
        if len(positions) > 0:
            return np.mean(positions, axis=0)
        else:
            return self.world.width/2.0, self.world.height/2.0

    def get_centroids(self, player=None, method='kmeans', num_clusters=-1,
                      iter=5, minit='points'):
        positions = self.get_positions(player)
        if len(positions) <= 10:
            for position in positions:
                yield position
        else:
            if method == 'kmeans':
                if num_clusters == -1:
                    # OPTION: INCREASE RANGE IF NECESSARY
                    for num_clusters in range(10, 0, -1):
                        try:
                            centroids, clustering = kmeans(positions,
                                                           min(num_clusters,
                                                               len(positions)),
                                                           iter,
                                                           minit="random",
                                                           missing="raise")
                        except:
                            continue
                        # print(num_clusters)
                        break
                else:
                    centroids, clustering = kmeans(positions,
                                                   min(num_clusters,
                                                       len(positions)),
                                                   iter,
                                                   minit=minit)
                for centroid in centroids:
                    yield centroid

    def get_squared_dist(self, x1, y1, x2, y2):
        return (x1 - x2)**2 + (y1 - y2)**2

    def get_dist(self, x1, y1, x2, y2):
        return math.sqrt(self.get_squared_dist(x1, y1, x2, y2))


    def new_move(self, action=None, group=0, x_range=(0.0, WORLD_WIDTH),
                 y_range=(0.0, WORLD_HEIGHT), vector=(0.0, 0.0), angle=0.0,
                 factor=0.0, max_speed=0.0, max_angular_speed=0.0,
                 vehicle_type=None, facility_id=-1, vehicle_id=-1):
        m = Move()
        m.action = action
        m.group = group
        m.left, m.right = x_range
        m.top, m.bottom = y_range
        m.x, m.y = vector
        m.angle = angle
        m.factor = factor
        m.max_speed = max_speed
        m.max_angular_speed = max_angular_speed
        m.vehicle_type = vehicle_type
        m.facility_id = facility_id
        m.vehicle_id = vehicle_id
        return m
