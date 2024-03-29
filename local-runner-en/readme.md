# Local Runner Server

## Startup parameters

1. One or more "*.properties" file names ordered by priority descending.
    At least one of these files should exist.
2. An Int64 number, that game engine will use to initialize the generator of random numbers.
    If not specified, the value of "seed" property will be used.
    If the "seed" property is also not specified, the generator will be initialized using the current time.

## Playback control

- **Space** - pause/play.
- **Up/Down** - speed up/slow down.
- **Right** - show next frame (only when paused).
- **H** - switch display mode.
- **Ctrl + +/-/0** - change scale.
- **Tab** - minimap on/off.
- **Ctrl + C** - switch unit colors.
- **F1-F2** - follow the corresponding player.

## Keyboard player

- **LMB (hold)** - select units (clear existing selection)
- **Shift + LMB (hold)** - add units to selection
- **Ctrl + LMB (hold)** - deselect units
- **LMB (double click on unit)** - select all nearby units of the same type (clear existing selection)
- **LMB (double click on factory)** - cyclically switches the type of producing vehicles
- **Shift + LMB (double click on unit)** - add nearby units of the same type to selection (or remove from selection, if the clicked unit is selected)
- **RMB** - move selected units
- **RMB (hold)** - rotate selected units relative to average X and Y of these units MMB (hold) or WSAD - drag camera
- **1-9** - select units of the corresponding group (clear existing selection)
- **Ctrl + 1-9** - make new group from selected units
- **Shift + 1-9** - add selected units to the corresponding group (create if not exist)
- **Ctrl + N** - enter/exit tactical nuclear strike mark mode (LMB - performs nuclear strike request while in mark mode), you should select a vehicle to be able to enter mark mode

## Game results file:

1. First line: game verdict.
    - **OK** - the game was successfully tested.
    - **FAILED** - got unexpected error while testing the game (for example, the strategy connection port 31001 is already in use).
2. Second line: the `"SEED"` keyword and the number used to initialize the generator of random numbers, for example, `"SEED 126522099021038"`.
3. Third and each next line: the result of participation of one strategy in ascending order of the strategy index.
    - Record format: `<place> <score> <verdict>`, for example, `"1 3576 OK"` or `"4 0 CRASHED"`.
