# 3dsrecomp

A static recompiler for **Nintendo 3DS** games that produces native **macOS** applications (Apple Silicon).
The game's ARM11 code is translated to C ahead of time and compiled for the Mac; the 3DS operating system
(Horizon kernel and services), the PICA200 GPU, the DSP audio and the input devices are replaced by host
implementations. It is not an emulator: the game runs as a native program.

First target: **Super Mario 3D Land** (EUR, `CTR-P-AREP`, title 0004000000053F00): it boots, plays its music,
shows the first-boot dialogs, the title screen, creates and loads save files, plays the intro and runs the game.

**New Super Mario Bros. 2** (USA, `CTR-P-ABEE`, title 000400000007AE00) also runs: first-boot dialogs, title
screen, the opening, the world map and World 1-1 at 60 fps with music and sound (built with `-DGAME=nsmb2` from
`data/nsmb2`; it needs the Mii data system archive, see `tools/import_system_archives.py` below). Coin Rush and
the add-on content are not supported (the DLC service is a stub).

**Tomodachi Life** (USA, `CTR-P-EC6E`, title 000400000008C300) runs too: save creation, date and time setup,
naming the island, making a look-alike Mii (imported through the Mii selector, then the game's own editor with
profile, text-to-speech voice and personality), the island and the apartment, at 60 fps with music and voices
(built with `-DGAME=tomodachi` from `data/tomodachi`). It needs the shared font and Mii data system archives
(`tools/import_system_archives.py`). With the open-source replacement archives, Mii faces show as a red "no"
placeholder; a real console's Mii data archive would be needed for the actual faces. Names are typed with the
Mac's keyboard (see [Typing](#typing)); StreetPass, SpotPass, QR codes and the camera are not supported.

**Metroid: Samus Returns** (USA, `CTR-P-A9AE`, title 00040000001BB200) runs: title screen, file select, the
story intro and the landing cutscene, then the first area of SR388 with Samus running, jumping, shooting, firing
missiles and taking damage, the bottom-screen map and HUD, Game Over / Continue, with music and sound (built with
`-DGAME=metroid` from `data/metroid`; extract it with `--no-unpack`, its RomFS is 700 MB). It is the heaviest of
the four on the GPU: on an M4 Pro the automatic resolution settles at 2x in gameplay (the title screen keeps 4x),
where it runs at full speed at about 85% of one core. amiibo, StreetPass and the New 3DS extras (faster CPU,
extended memory) are not used: the runtime presents an old 3DS, which the game supports.

**Animal Crossing: New Leaf – Welcome amiibo** (USA, `CTR-P-EAAE`, title 0004000000198E00) runs: title screen,
a new game with the train ride (Rover's questions, the name and town name on the software keyboard, the town
map choice), the arrival in town with Isabelle, the town hall, Main Street, Nook's Homes and placing the house
with Tom Nook, at full speed with music and sound (built with `-DGAME=acnl` from `data/acnl`). Most of its code
is in CRO modules (relocatable code loaded from the RomFS by `ldr:ro`), which the recompiler also translates:
see [CRO modules](#cro-modules-ldrro). The village's map and menus on the bottom screen work; saving was not
reached in tests (the game allows it only after the first day's tutorial). amiibo, StreetPass, SpotPass, QR codes
and online play are not supported: the services report nothing there.

Homebrew (`.3dsx`, devkitARM + libctru) goes through the same pipeline: a libctru test app (framebuffer graphics,
console on the bottom screen, sdmc file reading, input) runs, see [Homebrew](#homebrew-3dsx).

> No Nintendo files are included in this repository: no game code, no assets, no keys, no generated C. You need
> your own decrypted dump. The game code is recompiled on your machine from your own files.

## Requirements

Apple Silicon Mac, Xcode command line tools, Homebrew `llvm` (clang), `cmake`, `ninja`, `sdl3`, Python 3.

```sh
brew install llvm cmake ninja sdl3
python3 -m venv .venv && .venv/bin/pip install capstone unicorn   # only for tools/test_cpu.py and tools/disasm.py
```

## Building a game

1. **Extract** a decrypted `.cci`/`.3ds` (or `.cxi`) dump into `data/<name>/`:

   ```sh
   python3 tools/extract.py "Super Mario 3D Land.cci" data/sm3dl
   ```

   This writes `exheader.bin`, the decompressed `code.bin`, `romfs.bin` (the RomFS image the game reads) and an
   unpacked `romfs/` for inspection (`--no-unpack` skips it, which halves the disk space of large games).

2. **Recompile** the code to C (`generated/<name>/`, a few seconds):

   ```sh
   tools/regen.sh data/sm3dl sm3dl
   ```

3. **Build** the runtime and the game code (`build/sm3dl`):

   ```sh
   cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DGAME=sm3dl
   ninja -C build
   ```

4. **Play**: `./build/sm3dl data/sm3dl`, or make an app bundle with `tools/bundle_macos.sh sm3dl data/sm3dl`.

Saves go to `save/<title id>/` (`~/Library/Application Support/3dsrecomp/<title id>/` for app bundles). The game
saves by itself, as on a 3DS: its save archive is the `savedata/` folder there (and `extdata/`, if it uses extra
data); `backups/` holds copies made from the in-game menu and at each launch, and `settings.ini` the menu's
settings. The SD card and shared system data are `sdmc/` and `shared/` next to the title folders.

## In-game menu

Esc (or the gamepad's Home button) pauses the game and opens the menu; arrows or the d-pad choose, Return or
A selects, left/right change a value, Esc or B goes back. The mouse works too.

- **Saves**: the current save's size and date, back up now, automatic backup at each launch (the last 5 are kept),
  restore or delete a backup, delete the save (a copy is kept), show the folder in Finder. Restoring or deleting
  restarts the game, since it keeps its save in memory.
- **Clock**: the 3DS date and time, kept as an offset from the Mac's clock (year, month, day, hour, minute).
- **Graphics**: internal resolution (auto or 1-4x, live), anti-aliasing and texture filtering (after a restart),
  screen layout (the six F2 layouts), fullscreen, and **separate windows**: the top screen in the main window and
  the bottom screen in a second one, so each can go on its own display (F11 makes the window in use fullscreen;
  closing the bottom screen's window goes back to one window). Window positions are remembered per game.
- **Speed & audio**: the game's frame rate, game speed (50-300%; music follows), a frame rate counter, volume.
- Restart game, Quit.

Settings are per game (`settings.ini`); an environment variable given at launch (`GPU_MSAA`, `GPU_ANISO`) wins.

Optional: some games read system archives (Mii data, shared font). Open-source replacements can be imported from
an Azahar source checkout with `python3 tools/import_system_archives.py <azahar dir>` (writes `system_archives/`).

### Homebrew (.3dsx)

```sh
python3 tools/extract.py app.3dsx data/app          # relocates it to 0x00100000, synthesises an exheader
tools/regen.sh data/app app
cmake -S . -B build-app -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DGAME=app
ninja -C build-app && ./build-app/app               # BUILD=build-app tools/bundle_macos.sh app data/app
```

The app starts like a title launched by the system (libctru's `__service_ptr` stays 0, so it uses `srv:`/APT
and the heap sizes of its `_prm` block), with `argv[0] = sdmc:/3ds/<file>.3dsx` (`--argv0` to change it) so its
working directory is what the Homebrew Launcher gives it. The SD card is `save/sdmc/` (app bundles:
`~/Library/Application Support/3dsrecomp/sdmc/`). The title id is synthetic, in the homebrew range
(`0004000000Fxxx00`, from a hash of the SMDH title and author).

## Controls

| 3DS | Keyboard | Gamepad (by position) |
|---|---|---|
| Circle pad | Arrow keys | Left stick |
| D-pad | I J K L | D-pad |
| A / B / X / Y | X / Z / S / A | East / South / North / West |
| L / R | Q / W | Shoulders |
| ZL / ZR | 1 / 2 | Triggers |
| Start / Select | Enter / Backspace | Start / Back |
| Touch screen | Mouse on the bottom screen | |

F11 toggles fullscreen, F2 cycles the screen layout (stacked, top screen only, side by side, large top with a small bottom, bottom screen
only, stacked with the bottom screen first; the window takes the layout's shape and the mouse touches the bottom
screen wherever it is), Tab (or a click of the gamepad's right stick) swaps the screens in the single-screen and
stacked layouts, F3 cycles the internal
resolution (auto, 1x to 4x; Metal), F12 writes the state of every game thread to the log (press it if a
game seems stuck, then send `log.txt`), Esc opens the in-game menu (Quit is there, or Cmd+Q).

### Typing

- When a game opens the 3DS's system keyboard (names in Tomodachi Life, ...), a text field appears over the
  game: type with the Mac's keyboard (any language and input method, Cmd+V pastes), Return (or the gamepad's A)
  confirms and Esc (B) cancels when the game allows it. The game's length limit and character filters apply.
- Games that draw their own keyboard on the bottom screen (Animal Crossing: New Leaf's names and town name) can
  be typed into too when their profile describes it: while that keyboard is up, the Mac's keyboard types by
  tapping its keys (capitals use its Shift key; Backspace deletes, Return taps Confirm). The keys mapped to 3DS
  buttons type instead of pressing buttons until the keyboard closes; the arrows and the gamepad still play.

### Playing on one screen

In the "top screen only" and "bottom screen only" layouts (menu: Graphics):
- **Other screen in a corner**: the hidden screen appears small in a corner for a few seconds whenever its
  picture changes a lot (a dialog, a menu), or always. Clicking it touches the bottom screen; while the pointer is
  over it, a tab next to it moves it and a handle (or the scroll wheel) resizes it.
- **Automatic screen**: games with a profile (below) switch to the screen that matters, e.g. the bottom screen on
  menus that are played there. Tab shows the other screen until the game moves on.
- **Extra HUD**: games with a profile get the bottom screen's information during play (stored item, star coins,
  ...) drawn over the top screen with the game's own icons; E (or a click of the gamepad's left stick) uses the
  stored item by tapping the bottom screen where the game expects it.

### Game profiles

`games/<title id>/profile.txt` describes, for one game, values in its memory (lives, the stored item, which
scene it is in, ...), which screen matters when, and the HUD built from them. The format is documented at the top
of `runtime/profile.cpp`; it holds only addresses, numbers, RomFS paths and coordinates. The HUD's icons are
read from the game's own files at run time (BCLIM textures in NARC or SARC archives, Yaz0-compressed or not).
A profile can also map the game's own on-screen keyboard (`keyboard when ...`, `key ...`, see
[Typing](#typing)). To write one: `RECOMP_MEMDUMP` snapshots memory for diffing, `RECOMP_PROFILE=<file>` loads a draft and
`RECOMP_PROFILE_TRACE=<n>` logs every value every n frames.

## How it works

- `tools/extract.py` parses NCSD/NCCH, decompresses the ExeFS `.code` and strips the RomFS IVFC levels; for a
  `.3dsx` it applies the relocation tables and writes the same files (exheader, code.bin, SMDH as the icon).
- `tools/recomp/` is the recompiler (Python):
  - `arm.py` / `thumb.py` decode ARMv6K ARM and Thumb instructions (including VFPv2 and the media
    instructions) and lower each one to C. Guest registers become C locals; flags are separate 0/1 locals.
  - `gen.py` discovers functions (entry point, `bl` targets, pointers in data/rodata/literal pools,
    pc-relative code pointers, RVCT init arrays), resolves jump tables (`ldr pc, [pc, rX, lsl #2]`,
    `add pc, pc, rX, lsl #2`, GCC's byte tables `ldr[s]b rX, [table, rY]` + `add pc, pc, rX, lsl #2`, and
    RVCT's Thumb tables of byte/halfword offsets after `add rX, pc` / `ldrb rX, [rX, #k]` / `add pc, rX`) and
    emits one C function per guest function. `udf` traps (`__builtin_trap`) count as code, not data. Calls are C calls followed by
    a check that the callee returned where expected; returns to anywhere else unwind to a dispatcher that
    resumes at the right instruction (every call return site is a resumable entry). Tail branches and
    indirect jumps are `musttail` calls through the dispatcher.
  - `tools/test_cpu.py` differentially tests the lowering against Unicorn with random instruction blocks.
- `runtime/` replaces the 3DS:
  - `core.cpp`: a 4 GiB host reservation mirrors the guest address space (`g_mem + addr`), the dispatcher.
  - `kernel.cpp`: handles, threads (one host pthread per guest thread, but a single runner at a time with
    strict priority scheduling like the 3DS application core; when an interrupt readies a higher-priority
    thread, the running one gives way at its next loop branch), events, mutexes, semaphores, timers, address
    arbiters, shared memory, memory management, and the SVC interface.
  - `service.cpp` + `services/`: IPC to HLE services: srv, APT, fs (RomFS from `romfs.bin`, saves in host
    folders), gsp::Gpu (interrupt relay queue, GX command queue, framebuffers), hid, cfg, dsp, ptm, ac, frd,
    ldr:ro (CRO modules, below), and stubs for the rest.
  - `gpu/`: the PICA200 GPU, either on **Metal** (default) or in a multithreaded **software** rasterizer that
    is the reference implementation. With Metal, vertex programs are translated to compute kernels, one
    per program and set of bool/int uniforms (`shader_msl_spec()`: the uniform-dependent ifs, calls and loop
    counts are resolved when the kernel is generated, leaving structured code; programs whose flow is not
    properly nested use `shader_msl()`, a program-counter switch that reproduces the interpreter exactly), run
    as one dispatch per kernel per batch of draws; clipping,
    culling and snapping happen in a shared vertex shader, and the fragment pipeline (TEV combiners, fragment
    lighting, fog, tests, blending) becomes MSL generated per pipeline state, rendering at an internal resolution
    scale with guest memory kept authoritative by lazy write-back. Geometry shaders, immediate-mode vertices and
    vertex programs still compiling run in the CPU interpreter (pre-decoded, 4-wide SIMD).
  - `audio/`: HLE of the DSP audio firmware, output through SDL3.
  - `frontend.cpp`: SDL3 window, input, VBlank pacing.

### CRO modules (ldr:ro)

Some games keep most of their code in CRO modules: relocatable files in the RomFS (`*.cro`, with `static.crs`
describing the main program's exports and imports) that the game reads into its heap and hands to `ldr:ro`,
which relocates and links them; where they land changes from run to run. Recompiled code needs fixed addresses,
so `tools/recomp/cro.py` gives every module a fixed *slot* (from 0x40000000, in name order, listed in the
generated `guest_cros[]`), links all modules there for the analysis and recompiles their code with the program.
At run time `runtime/services/ro.cpp` implements the loader (rebasing the header tables, internal relocations,
imports and exports by name, index and segment offset, the auto-link list, fix levels, unload and unlink) with
one change: each module's code and read-only data are copied to its slot and relocated against it, so every code
pointer the game ever sees (relocations, exports, return addresses) is a slot address the recompiled code knows.
Its .data and .bss stay where the game put them. Words the loader patches at run time (relocation targets in
modules, and the main program's words that static.crs imports point at module symbols) are never folded as
constants by the recompiler. `tools/disasm.py` shows module code at slot addresses.

## Graphics options

- `GPU_BACKEND=metal|sw` (default `metal`).
- `GPU_SCALE=1..4`: internal resolution of the Metal backend, in multiples of the 3DS screens. Without it the
  scale follows the window: the smallest one that covers the window's pixels (4x for the default window on a
  Retina display, 2x on a standard one), re-chosen when the window is resized. F3 cycles auto, 1x to 4x.
  In auto mode the scale is also lowered, one step at a time, when the game drops frames (its frame rate is
  neither ~60 nor a steady ~30) while the GPU is busy over half the time, and raised again when the next scale
  should fit; heavy games (Metroid: Samus Returns) settle at 2x on an M4 Pro, others keep 4x.
- `GPU_MSAA=1|2|4|8`: multisample anti-aliasing of the Metal render targets (default 4).
- `GPU_ANISO=1..16`: anisotropic filtering of the textures the game filters linearly with mipmaps (default 16).
- `GPU_TEX_FILTER=0|1|2`: texture upscaling (default 0, off; `gpu::set_texture_filter` at run time). 1 (Sharp)
  is an edge-directed 2x pass in the spirit of xBR, applied twice for 4x: hard steps stay crisp, diagonal and
  curved staircases become straight edges, gradients and anti-aliased edges are interpolated. 2 (Smooth) is an
  edge-adaptive Lanczos-2 resampler in the spirit of FSR1's EASU (kernel stretched along edges, clamped to the
  nearest texels): smooth curves and crisp edges, best for anti-aliased art such as NSMB2's tiles. Both run as
  Metal compute kernels (`runtime/gpu/texfilter.mm`) once per decoded texture, 4x above internal scale 2 (else
  2x), mix colours weighted by alpha (no dark fringes at cut-outs), follow the texture's wrap mode (no seams on
  tiled textures) and keep the game's mip levels below the upscaled ones. A texture the game rewrites in parts
  (NSMB2's tile atlas with its animated coins and ? blocks) only has the changed 8x8 tiles redone; one rewritten
  entirely over and over is left alone. Not upscaled: render-to-texture data (render targets and copies of
  them), cube and shadow maps, HILO8, textures over 1024x1024 and thin or tiny lookup tables (under 256 texels,
  or 8:1 and thinner); upscaled textures use at most 512 MiB.
- The window is presented in step with the display's refresh (no tearing) at normal speed; `RECOMP_NOVSYNC=1`
  turns that off. Above 100% speed presents never wait for the display.
- The screens are drawn into the window with nearest filtering at integer magnifications, sharp bilinear
  filtering at other magnifications and linear filtering when the image is larger than its place in the window.
- `GPU_THREADS=n` worker threads for the software backend.
- `GPU_CPU_VS=1`: shade vertices with the CPU interpreter instead of Metal compute kernels.

### Shader precompilation (Metal)

Each PICA fragment configuration becomes its own Metal pipeline, and each vertex program a compute kernel. To
keep the Metal compiler out of gameplay:

- `games/<title id>/pipelines.bin` lists the pipelines a game uses. The build turns their fragment shaders into
  `build/shaders.metallib` (`<game> --emit-shaders` + `tools/build_metallib.sh`, about 2 s, skipped when
  `xcrun metal` is missing); `tools/bundle_macos.sh` copies both into the app.
- At startup every known pipeline is created on background threads from the precompiled library (a few ms
  each), before the game needs it.
- `games/<title id>/vsprograms.bin` (optional, not in the repository: it holds the game's own vertex shader
  programs, so it is made locally from your own play, see below) lists the vertex programs (with the uniforms
  each kernel is specialized for); their kernels go into the same library. Without it, vertex kernels are
  compiled in the background on first use while the CPU shades those draws, so nothing is skipped.
- Pipelines the list misses are recorded in `<saves>/shadercache/<title id>.bin` (prebuilt in the background
  on later launches) and compiled in the background the first time; their draws are skipped for those few
  frames, and the screen keeps showing the previous complete frame meanwhile. `GPU_SYNC_SHADERS=1` compiles them on the spot instead (the old behaviour). Vertex programs the list
  misses (`<saves>/shadercache/<title id>.vs.bin`) are compiled in the background too, the CPU shading their
  draws meanwhile, so nothing is skipped.
- To extend a game's lists, play with fresh saves and merge the caches:
  `tools/merge_pipelines.py games/<title id>/pipelines.bin games/<title id>/pipelines.bin save/shadercache/<title id>.bin`
  (and `tools/merge_pipelines.py games/<title id>/vsprograms.bin save/shadercache/<title id>.vs.bin` to make
  your local vertex program list; git ignores it). The shipped `pipelines.bin` files only hold GPU state keys
  (which fixed-function modes a draw uses), no game code or data.
- Diagnostics: `GPU_HITCH_MS=<ms>` logs command lists slower than that; `GPU_VERIFY_KEYS=1` checks that every
  new pipeline key decodes back to the same shader; `GPU_NO_METALLIB=1` ignores the precompiled library;
  `GPU_NO_PRECOMPILE=1 GPU_SYNC_SHADERS=1` turns all of this off for comparisons.

## Debugging aids

- `RECOMP_VERBOSE=1|2` (or `-v`/`-vv`): kernel and IPC logs.
- `RECOMP_INPUT` also takes host input for testing the frontend: `<frame>:KEY_<SDL key name>`, typed text
  `<frame>:TEXT_<text>` (no `,` or `:`), `<frame>:MOUSE_DOWN@x/y` (`MOUSE_MOVE`, `MOUSE_UP`) and `<frame>:MOUSE_WHEEL@x/y/dy`, in layout coordinates.
- `RECOMP_TRACE=n`: log the first n dispatcher transitions.
- `RECOMP_MEMDUMP=<frame>,...`: snapshots of guest memory at those frames (`mem_<frame>_<address>.bin`: the
  data/bss segment and every mapped region), for finding where a game keeps its state by diffing them.
- `RECOMP_DUMP=<frame>`: dump that frame's GX work (memory + operations) for the GPU replay tool.
- `RECOMP_SPEED=<percent>`: game speed for this run (the menu's Game speed, up to 2000%). The game's clock
  (timers, sleeps, the system tick) runs at the same pace, so scripted runs reach the same point on the same
  frame at any speed the Mac keeps up with: 400% plays SM3DL's intro into World 1-1 in 35 s instead of 135 s.
  Above about 500% the game falls behind (audio emulation, most likely) and scripts drift.
- `RECOMP_SHOT=<n>`: write both screens as PPM every n frames (from frame `RECOMP_SHOT_FROM`, if set).
  `RECOMP_FRAMES=<n>`: quit after n frames.
  `RECOMP_WINSHOT=<f1,f2,...>`: the whole window (menu included) at those frames.
- `RECOMP_INPUT="frame:BUTTONS:duration,..."`: scripted input (e.g. `600:START:10,900:A:5`; `T@x/y` touches the
  bottom screen at pixel x, y, e.g. `700:T@160/222:6`).
  Host keys for the menu: `500:KEY_Escape,530:KEY_Down,560:KEY_Return` (SDL key names).
  When building the string in zsh, write `${f}:A`, not `$f:A`: zsh's `:A` modifier turns `$f` into an absolute path.
- Every run writes its log to `<save dir>/<title id>/log.txt` (the previous one is kept as `log.txt.prev`); the
  app bundle shows fatal errors in a dialog that points there.
- Code the recompiler did not find is recompiled on the fly when the source tree is present; otherwise the run
  stops. Either way the address is appended to `<save dir>/missing_functions.txt`, and `tools/regen.sh` folds it
  into `games/<title id>/functions.txt` (from `save/` and from the app's Application Support folder) and
  regenerates.
- `RECOMP_ROMFS_SWAP="A=B,..."`: every RomFS file whose path contains `A` is served with the data of the file
  whose path has `B` instead, e.g. `RECOMP_ROMFS_SWAP=/FirstStage=/CaveInkStage` loads another level in place
  of World 1-1, which is how levels past 1-1 are tested without a completed save.
- `RECOMP_ROMFS_REPLACE="<path>=<host file>,..."`: serve a file from the Mac in place of a RomFS file (mods,
  edited levels for tests).
- If a game submits no GPU work for 5 s, the log records the state of every thread (`[diag]`), each with the
  chain of game functions it is in (`in <innermost> < <caller> < ...`); outside `RECOMP_STATS` runs a stats line
  is logged every 30 s.
- `RECOMP_REGION` / `RECOMP_LANGUAGE`: the console region (0 JPN, 1 USA, 2 EUR, ...) and language (0 JP, 1 EN, ...)
  the system settings report. The region defaults to the game's own, from its icon's region lockout.
- `RECOMP_FS_DELAY=0`: RomFS reads return at once instead of taking as long as on a 3DS (the reading thread
  sleeps for the game card's latency and transfer time, as games that load in the background expect).
- `RECOMP_SWKBD_TEXT`: the system keyboard "types" this at once instead of asking (empty: the game's initial
  text, else "Player"), for unattended tests.
- `RECOMP_LIVE=<file>`: input for a running test, one command per line appended to the file while the game runs
  (`<buttons> <frames>` with the button names of `RECOMP_INPUT`, e.g. `A 6`, `CDOWN+B 60`, `T@160/120 8`, or
  `WAIT <frames>`); commands run one after another, each followed by a few released frames, and the log says when
  each starts. A line `DUMP` writes the state of every thread to the log. With `RECOMP_SHOT`, this drives long
  explorations step by step without replaying a script from boot.
- A jump to a null function pointer stops with the registers in the log (instead of trying to recompile it), and
  `RECOMP_WATCH` reports which code touched its page before a fatal error.
- `tools/disasm.py <gamedir> <addr> [count] [thumb]` disassembles guest code (including CRO modules).

## License

The recompiler, runtime and tools are licensed under the [European Union Public Licence v1.2](LICENSE) (EUPL-1.2).
This covers this repository only: the games, and anything extracted or generated from them, remain their owners' and
are not part of it.
