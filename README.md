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
placeholder; a real console's Mii data archive would be needed for the actual faces. Names typed on the software
keyboard come from `RECOMP_SWKBD_TEXT` (default "Player"); StreetPass, SpotPass, QR codes and the camera are not
supported.

**Metroid: Samus Returns** (USA, `CTR-P-A9AE`, title 00040000001BB200) runs: title screen, file select, the
story intro and the landing cutscene, then the first area of SR388 with Samus running, jumping, shooting, firing
missiles and taking damage, the bottom-screen map and HUD, Game Over / Continue, with music and sound (built with
`-DGAME=metroid` from `data/metroid`; extract it with `--no-unpack`, its RomFS is 700 MB). It is the heaviest of
the four on the GPU: on an M4 Pro the automatic resolution settles at 2x in gameplay (the title screen keeps 4x),
where it runs at full speed at about 85% of one core. amiibo, StreetPass and the New 3DS extras (faster CPU,
extended memory) are not used: the runtime presents an old 3DS, which the game supports.

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
screen wherever it is), F3 cycles the internal
resolution (auto, 1x to 4x; Metal), F12 writes the state of every game thread to the log (press it if a
game seems stuck, then send `log.txt`), Esc opens the in-game menu (Quit is there, or Cmd+Q).

## How it works

- `tools/extract.py` parses NCSD/NCCH, decompresses the ExeFS `.code` and strips the RomFS IVFC levels; for a
  `.3dsx` it applies the relocation tables and writes the same files (exheader, code.bin, SMDH as the icon).
- `tools/recomp/` is the recompiler (Python):
  - `arm.py` / `thumb.py` decode ARMv6K ARM and Thumb instructions (including VFPv2 and the media
    instructions) and lower each one to C. Guest registers become C locals; flags are separate 0/1 locals.
  - `gen.py` discovers functions (entry point, `bl` targets, pointers in data/rodata/literal pools,
    pc-relative code pointers, RVCT init arrays), resolves jump tables (`ldr pc, [pc, rX, lsl #2]`,
    `add pc, pc, rX, lsl #2`, and GCC's byte tables `ldr[s]b rX, [table, rY]` + `add pc, pc, rX, lsl #2`) and
    emits one C function per guest function. `udf` traps (`__builtin_trap`) count as code, not data. Calls are C calls followed by
    a check that the callee returned where expected; returns to anywhere else unwind to a dispatcher that
    resumes at the right instruction (every call return site is a resumable entry). Tail branches and
    indirect jumps are `musttail` calls through the dispatcher.
  - `tools/test_cpu.py` differentially tests the lowering against Unicorn with random instruction blocks.
- `runtime/` replaces the 3DS:
  - `core.cpp`: a 4 GiB host reservation mirrors the guest address space (`g_mem + addr`), the dispatcher.
  - `kernel.cpp`: handles, threads (one host pthread per guest thread, but a single runner at a time with
    strict priority scheduling like the 3DS application core), events, mutexes, semaphores, timers, address
    arbiters, shared memory, memory management, and the SVC interface.
  - `service.cpp` + `services/`: IPC to HLE services: srv, APT, fs (RomFS from `romfs.bin`, saves in host
    folders), gsp::Gpu (interrupt relay queue, GX command queue, framebuffers), hid, cfg, dsp, ptm, ac, frd,
    and stubs for the rest.
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
  frames. `GPU_SYNC_SHADERS=1` compiles them on the spot instead (the old behaviour). Vertex programs the list
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
- `RECOMP_TRACE=n`: log the first n dispatcher transitions.
- `RECOMP_DUMP=<frame>`: dump that frame's GX work (memory + operations) for the GPU replay tool.
- `RECOMP_SHOT=<n>`: write both screens as PPM every n frames. `RECOMP_FRAMES=<n>`: quit after n frames.
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
- If a game submits no GPU work for 5 s, the log records the state of every thread (`[diag]`); outside
  `RECOMP_STATS` runs a stats line is logged every 30 s.
- `RECOMP_REGION` / `RECOMP_LANGUAGE`: the console region (0 JPN, 1 USA, 2 EUR, ...) and language (0 JP, 1 EN, ...)
  the system settings report. The region defaults to the game's own, from its icon's region lockout.
- `RECOMP_SWKBD_TEXT`: what the software keyboard applet "types" (default: the game's initial text, else "Player").
- `tools/disasm.py <gamedir> <addr> [count]` disassembles guest code.
