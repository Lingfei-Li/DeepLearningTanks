
class Const:
    # direction constants
    (DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT) = range(4)

    # action
    (DO_FIRE, DO_UP, DO_RIGHT, DO_DOWN, DO_LEFT, DO_NOTHING) = range(6)

    # bullet's stated
    (STATE_REMOVED, STATE_ACTIVE, STATE_EXPLODING) = range(3)

    (OWNER_PLAYER, OWNER_ENEMY) = range(2)

    ACTIONS = range(0,6)
