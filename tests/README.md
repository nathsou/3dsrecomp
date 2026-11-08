# Game regression tests

`tools/gametest.py` runs each game headless with scripted input and checks what it shows and logs. Run it after
changing anything shared (the runtime, the recompiler, a GPU backend):

```sh
python3 tools/gametest.py --checkpoints <folder of your Checkpoint backups>   # every test that can run here
python3 tools/gametest.py 0004000000049100            # one game (title id, test name, or <title id>-<test>)
python3 tools/gametest.py --list                       # what can run here, and why not
python3 tools/gametest.py -j 1                         # one at a time: frame rates are checked too
```

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
  picture exists, the screens are close to it (mean difference of 40x24 thumbnails below `ref_tolerance`, 24 by
  default: timing differs a little between runs).
- `fps`: the game's frame rate over a stretch of frames (from the `[stats]` lines), only with `-j 1`.

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
