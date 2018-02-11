#pragma once

#ifndef _MY_STRATEGY_H_
#define _MY_STRATEGY_H_

#include "Strategy.h"

#include <iostream>
#include <map>

class MyStrategy : public Strategy {
private:
    model::Player *me, *en;
    model::World *world;
    model::Game *game;
    model::Move *move;

    std::map<int, model::Facility> allFacilities;
    std::map<int, model::Vehicle> allVehicles;
public:
    MyStrategy();

    void move(const model::Player& me, const model::World& world,
              const model::Game& game, model::Move& move) override;
    void update(model::Player& me, model::Player& en,
                model::World& world, model::Game& game,
                model::Move& move);
    
};

#endif
