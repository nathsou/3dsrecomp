# Game regression tests

`tools/gametest.py` runs each game headless with scripted input and checks what it shows and logs. Run it after
changing anything shared (the runtime, the recompiler, a GPU backend):

```sh
python3 tools/gametest.py --checkpoints <folder of your Checkpoint backups>   # every test that can run here
python3 tools/gametest.py 0004000000049100            # one game (title id, test name, or <title id>-<test>)
python3 tools/gametest.py --list                       # what can run here, and why not
python3 tools/gametest.py --twice                      # each test twice: the pictures must be the same
python3 tools/gametest.py --realtime -j 1              # at the 3DS's speed, one at a time: frame rates are checked
```

Tests run in deterministic mode (`RECOMP_DETERMINISTIC=1`): the same run gives the same pictures, byte for byte, on
every run and under any load, as fast as the machine goes (several times the 3DS's speed). Guest time is virtual: it
passes as guest code runs (a fixed amount per loop iteration) and jumps ahead when every guest thread waits; VBlank,
input and audio come at fixed guest times; GPU work is finished when the game's call that queued it returns; the
date is 2026-01-01; shaders are compiled before the draws that need them, at 1x resolution. `--realtime` runs the
game as you play it, with real timing.

A test needs the game's files in `data/<name>/` (any name: they are found by title id) and a build of them
(a `build*/` folder configured with `-DGAME=<name>`). Tests that start from a save need your own Checkpoint backup
(`--checkpoints`, or `$RECOMP_TEST_CHECKPOINTS`; the newest backup of the title is used). Missing pieces skip the
test. Nothing from the games goes in the repository: logs and pictures are written to
`shots/tests/<title id>-<test>/` (with `sheet.png`, the checkpoints side by side), and reference pictures, if you
make them, to `~/.local/share/3dsrecomp/test-refs/` (`--update-refs`; `--refs` or `$RECOMP_TEST_REFS` elsewhere).

## Checks

- The run ends normally: exit code 0, no fatal error in the log.
- No code the recompiler missed was recompiled on the fly (the addresses are then in `save/missing_functions.txt`
  of the run's folder; `tools/regen.sh` adds them). `allow_lazy = true` turns this off.
- Every `[service] unimplemented command 0x...` line names a command the test lists in `known_unimplemented`: a game
  that starts using one more is noticed (implement it in `runtime/services/`, or add it to the list).
- At each checkpoint: the listed screens are not blank, the profile's values are as given, and, where a reference
  picture exists, the screens are the same as it, pixel for pixel. Deterministic runs and real-time runs have
  references of their own (`<refs>/<test>/deterministic/` and `<refs>/<test>/`); a real-time run only has to be close
  to its reference (mean difference of 40x24 thumbnails below `ref_tolerance`, 24 by default: timing differs a little
  between runs). `ref_tolerance` set in a test applies to both.
- `--twice`: a second run gives the same pictures.
- `fps`: the game's frame rate over a stretch of frames (from the `[stats]` lines), only with `--realtime -j 1`.

## Format

`tests/games/<title id>/<test>.toml`:

```toml
game = "nsmb2"                       # (informative)
description = "..."
frames = 2400                        # the run's length
save = "checkpoint"                  # optional: start from the newest Checkpoint backup of the title
checkpoint_id = "0x00864"            # optional: the backup of another title (an earlier edition's save)
save_dirs = ["replay_"]              # optional: empty folders to make in the save (Checkpoint leaves them out)
swkbd = "Link"                       # optional: what the system keyboard types (RECOMP_SWKBD_TEXT)
input = ["A:6 every 300 from 600 to 2400", "1500:T@160/155:8"]   # RECOMP_INPUT events
known_unimplemented = ["boss:U 0x00250082"]
fps = { from = 4400, to = 5600, min = 29 }
env = { RECOMP_REGION = "2" }        # optional: more environment for the run

[[checkpoint]]
frame = 2400
screens = ["top", "bottom"]          # not blank
values = { scene = 18, focus = "both" }   # the profile's values (RECOMP_PROFILE_TRACE)
ref = true                           # compare with the reference picture, if there is one (default)
```

Input events are `RECOMP_INPUT`'s (`<frame>:<buttons>:<frames>`, `T@x/y` for the touch screen, host keys...).
`BUTTONS:N every E from F to T` repeats a press. To turn a session played with `RECOMP_LIVE` into a test, take the
frames at which its commands started from the log (`[live] command N starts`).

## Soak runs

`tools/soak.py` plays each game for a long time (30000 frames by default, about 8 minutes) with random input from a
seed: buttons, the circle pad held in a direction, taps on the touch screen. It reports what went wrong, grouped by
kind: fatal errors, code the recompiler missed, unimplemented service commands, unsupported GPU features, a game
that stopped submitting GPU work. Runs are deterministic, so a run that finds something plays again exactly
(`input.txt` next to its log in `shots/soak/<name>-<seed>/`).

```sh
python3 tools/soak.py --seeds 1,2 --checkpoints <folder of your Checkpoint backups>
python3 tools/soak.py --frames 100000 nsmb2      # one game, longer
```
