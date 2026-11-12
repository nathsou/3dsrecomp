# 3dsrecomp

A static recompiler that turns **Nintendo 3DS** games into native applications for **macOS** (Apple Silicon),
**Linux** (x86-64, Vulkan) and **iOS/iPadOS**.

The game's ARM11 code is translated to C ahead of time and compiled for the host. The 3DS itself (the Horizon
kernel and services, the PICA200 GPU, the DSP audio, the input devices) is replaced by host implementations. It
is not an emulator: the game runs as a native program, rendered with Metal or Vulkan at up to 4x resolution.

> No Nintendo files are included in this repository: no game code, no assets, no keys, no generated C. You need
> your own dump; the game is recompiled on your machine from your own files.

## Games

| Game | Version | `-DGAME=` |
|---|---|---|
| Super Mario 3D Land | EUR | `sm3dl` |
| New Super Mario Bros. 2 | USA, EUR | `nsmb2` |
| Mario Kart 7 | EUR | `mk7` |
| The Legend of Zelda: Ocarina of Time 3D | EUR | `oot3d` |
| The Legend of Zelda: Majora's Mask 3D | EUR (eShop) | `mm3d` |
| Metroid: Samus Returns | USA | `metroid` |
| Animal Crossing: New Leaf – Welcome amiibo | USA, EUR (eShop) | `acnl` |
| Tomodachi Life | USA, EUR | `tomodachi` |
| Hey! Pikmin | EUR | `pikmin` |
| Picross 3D: Round 2 | EUR (eShop) | `picross` |
| Pullblox | EUR (eShop) | `pullblox` |
| Kid Icarus: Uprising | EUR | `kiu` |
| Star Fox 64 3D | EUR | `starfox` |
| The Legend of Zelda: A Link Between Worlds | EUR | `albw` |
| Kirby: Triple Deluxe | EUR | `kirbytd` |

All of them run at full speed with music and sound. Online play, StreetPass, SpotPass, amiibo, QR codes and the
camera are not supported. Homebrew (`.3dsx`, devkitARM + libctru) goes through the same pipeline.

Kid Icarus: Uprising's aiming, done with the stylus on a 3DS, also works with the right stick or the mouse.
Ocarina of Time 3D gets a free camera on the right stick, like Majora's Mask 3D's on the C-stick (Z-targeting puts
it back behind Link; the in-game menu turns it off or inverts it).

Some games (Tomodachi Life, NSMB2, Mario Kart 7, Pullblox) need the Mii data or shared font system archives.
Open-source replacements can be imported from an Azahar source checkout:
`python3 tools/import_system_archives.py <azahar dir>`.

## Building a game

**macOS:**

```sh
brew install llvm cmake ninja sdl3
```

**Linux** (Vulkan 1.3 GPU; Arch-based example):

```sh
sudo pacman -S clang cmake ninja sdl3 vulkan-icd-loader vulkan-headers shaderc freetype2 harfbuzz fontconfig curl ffmpeg
```

**In one step:** `python3 tools/port.py <dump> [<name>] [--app]` extracts the dump, recompiles it, builds it and,
with `--app`, makes an app of it (each step skipped when already done; keys are looked for next to the dump, in the
current folder and in Azahar's or Citra's `sysdata`). `--launcher` builds one program with a game list holding every
recompiled game (`build-all/recomp`, `cmake -DGAMES=all`; with `--app`, an app of it). The steps it runs:

1. **Extract** a decrypted `.cci`/`.3ds`/`.cxi` dump:

   ```sh
   python3 tools/extract.py "Super Mario 3D Land.cci" data/sm3dl
   ```

   Encrypted dumps and eShop `.cia` files are decrypted with your own console's keys: `--boot9 boot9.bin`, plus
   `--keys aes_keys.txt` for a CIA, and `--seeddb seeddb.bin` (or `--seed <hex>`) for titles using seed crypto.
   `--no-unpack` skips unpacking the RomFS for inspection, saving disk space on large games.

2. **Recompile** the code to C (a few seconds):

   ```sh
   tools/regen.sh data/sm3dl sm3dl
   ```

3. **Build**:

   ```sh
   cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DGAME=sm3dl
   ninja -C build
   ```

4. **Play**: `./build/sm3dl data/sm3dl`, or make an app with `tools/bundle_macos.sh sm3dl data/sm3dl`
   (Linux: `tools/bundle_linux.sh`, iOS: `tools/bundle_ios.sh`; `--games <name>...` for a multi-game build; see the
   scripts for options).

**Tests:** `python3 tools/gametest.py` plays every game you have through scripted runs and checks them
(`tests/README.md`). They run in deterministic mode (`RECOMP_DETERMINISTIC=1`): a run gives the same pictures every
time, faster than the 3DS.

Saves go to `save/<title id>/` (`~/Library/Application Support/3dsrecomp/` for app bundles,
`~/.local/share/3dsrecomp/` on Linux). A real console's save goes in its `savedata/` folder; Checkpoint backups leave
out empty folders, and Mario Kart 7 resets a save whose empty `replay_` folder is missing.

## Controls

| 3DS | Keyboard | Gamepad |
|---|---|---|
| Circle pad | Arrow keys | Left stick |
| D-pad | I J K L | D-pad |
| A / B / X / Y | X / Z / S / A | East / South / North / West |
| L / R | Q / W | Shoulders |
| ZL / ZR | 1 / 2 | Triggers |
| C-stick (Circle Pad Pro) | | Right stick |
| Start / Select | Enter / Backspace | Start / Back |
| Touch screen | Mouse | |

- **Esc** opens the in-game menu: saves and backups, cloud saves, clock, graphics, speed and volume.
- **F2** cycles the screen layout (stacked, side by side, top only, …), **Tab** swaps the screens,
  **F3** cycles the internal resolution, **F11** toggles fullscreen.
- When a game opens the system keyboard, you type with your real keyboard.

## Features

- **Upscaled rendering** on Metal or Vulkan: internal resolution up to 4x (chosen automatically from the window
  size and GPU load), MSAA, anisotropic filtering and optional texture upscaling. Shaders are precompiled so the
  shader compiler stays out of gameplay.
- **Separate windows**: the top and bottom screens can each go on their own display.
- **One-screen play**: games with a profile (`games/<title id>/profile.txt`) switch automatically to the screen
  that matters, show the hidden screen in a corner, and draw the bottom screen's information as an extra HUD.
  Animal Crossing: New Leaf plays entirely on the top screen, with item stacks in the pockets as in New Horizons.
- **Cloud saves** through your own S3-compatible bucket (Backblaze B2, Cloudflare R2, AWS S3, MinIO), shared
  between Macs, PCs, iPhones and iPads. Saves from a real 3DS (Checkpoint backups) can be uploaded with
  `tools/checkpoint_cloud.py`.
- **iPhone and iPad** builds with touch controls and gamepad support, one game per app or several in one.

## How it works

- `tools/extract.py` parses NCSD/NCCH/CIA/3DSX, decrypts and decompresses the code, and extracts the RomFS.
- `tools/recomp/` is the recompiler (Python). It decodes ARMv6K ARM and Thumb code (with VFPv2), discovers
  functions and jump tables, and emits one C function per guest function. Guest registers become C locals;
  unexpected returns and indirect jumps go through a dispatcher. Games that keep their code in CRO modules
  (relocatable code loaded at run time, e.g. Animal Crossing) have each module recompiled at a fixed address.
  `tools/test_cpu.py` differentially tests the generated code against Unicorn; `gen.py --verify-stores` makes the
  generated code check the registers it leaves out when it saves them around calls.
- `runtime/` replaces the 3DS:
  - `core.cpp`: a 4 GiB host reservation mirrors the guest address space.
  - `kernel.cpp`: threads, synchronization, memory and the SVC interface, with the 3DS's priority scheduling.
  - `services/`: HLE of the system services (fs, gsp, hid, apt, dsp, ldr:ro, …).
  - `gpu/`: the PICA200 on Metal, Vulkan, or a software reference rasterizer. Vertex programs become compute
    kernels and the fragment pipeline becomes generated shaders.
  - `audio/`: HLE of the DSP audio firmware.

Code the recompiler missed is recompiled on the fly when the source tree is present and recorded so that
`tools/regen.sh` includes it next time. Many `RECOMP_*` and `GPU_*` environment variables exist for debugging and
scripted testing (input scripts, screenshots, memory dumps, speed control); see `runtime/` for details. Two for
finding why a game hangs: `RECOMP_WATCH=<address>[+<bytes>]` logs every write to that guest memory with the code that
made it (Linux x86-64; a live script's `WATCH` arms it later), and `RECOMP_SCHEDLOG=all` logs every thread's waits
and wake-ups.

## License

The recompiler, runtime and tools are licensed under the [European Union Public Licence v1.2](LICENSE) (EUPL-1.2).
This covers this repository only: the games, and anything extracted or generated from them, remain their owners'.
