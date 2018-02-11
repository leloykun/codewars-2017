#include "MyStrategy.h"

#define PI 3.14159265358979323846
#define _USE_MATH_DEFINES

#include <map>
#include <cmath>
#include <cstdlib>

using namespace model;

MyStrategy::MyStrategy() { }

void MyStrategy::move(const Player& me, const World& world,
                      const Game& game, Move& move) {
    Player en = world.getOpponentPlayer();
    this.update(me, en, world, game, move);

    if (world.getTickIndex() == 1) {
        move.setAction(ActionType::CLEAR_AND_SELECT);
        move.setRight(world.getWidth());
        move.setBottom(world.getHeight());
        return;
    }

    if (world.getTickIndex() == 2) {
        move.setAction(ActionType::MOVE);
        move.setX(world.getWidth() / 2.0);
        move.setY(0);
        //move.setY(world.getHeight() / 2.0);
    }
}

void MyStrategy::update(Player& me, Player& en, World& world, Game& game,
                        Move& move) : me(en), en(en), world(world), game(game),
                        move(move) {
    
}

