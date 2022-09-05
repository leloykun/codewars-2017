from model.ActionType import ActionType
from model.Game import Game
from model.Move import Move
from model.Player import Player
from model.World import World

import numpy as np


EPS = 0.1
ALL_VEHICLE_TYPES = [0, 1, 2, 3, 4]
WORLD_WIDTH = 1024
WORLD_HEIGHT = 1024


class Utils:
    all_vehicles = {}
    all_facilities = {}
    all_groups = {}

    @staticmethod
    def init(strat):
        Utils.strat = strat
        Utils.me = strat.me
        Utils.en = strat.en

    @staticmethod
    def get_positions(units=None, player=None, types=ALL_VEHICLE_TYPES, group=0):
        if units is None:
            units = Utils.get_units(player, types, group)
        positions = []
        for unit in units:
            positions.append([unit.x, unit.y])
        return np.array(positions)

    @staticmethod
    def get_units(player, types=ALL_VEHICLE_TYPES, group=0,
                  x_range=[0, WORLD_WIDTH], y_range=[0, WORLD_HEIGHT]):
        units = []
        for id in Utils.all_vehicles:
            unit  = Utils.all_vehicles[id]
            if unit.player_id == player.id and unit.type in types and \
               (group == 0 or (group != 0 and group in unit.groups)) and \
               x_range[0] <= unit.x <= x_range[1] and \
               y_range[0] <= unit.y <= y_range[1]:
                units.append(unit)
        return units


class Group:
    def __init__(self, id, units):
        self.id = id
        self.units = units
        self.update()

    def update(self):
        self.num_units = len(self.units)
        self.positions = Utils.get_positions(self.units)
        self.vehicle_cnt = np.histogram([unit.type for unit in self.units], bins=5)
        self.vehicle_freq = self.vehicle_cnt / self.num_units

    def get_median(self):
        pass

    def get_centroid(self):
        pass


class MyStrategy:
    def init(self, me, en, world, game):
        self.me = me
        self.en = en
        self.world = world
        self.game = game
        Utils.init(self)

    def update(self, world, game):
        self.world = world
        self.game = game
        # Utils.update()
        # for group in Utils.all_groups:
        #     group.update().

    def move(self, me, world, game, move):
        if world.tick_index == 0:
            en = world.get_opponent_player()
            self.init(me, en, world, game)
        else:
            self.update(world, game)

        if world.tick_index == 1:
            move.action = ActionType.CLEAR_AND_SELECT
            move.right = world.width
            move.bottom = world.height

        if world.tick_index == 2:
            move.action = ActionType.MOVE
            move.x = world.width / 2.0
            move.y = world.height / 2.0
