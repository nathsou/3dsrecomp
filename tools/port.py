#!/usr/bin/env python3
"""Port a game in one step: extract the dump, recompile its code, build it, and (with --app) make an app of it.

  tools/port.py <dump> [<name>] [--app] [--launcher] [--force] [--jobs N]
  tools/port.py --launcher [--app]

  <dump>      a decrypted .cci/.3ds/.cxi, an encrypted one or a .cia (keys: boot9.bin, aes_keys.txt and, for titles
              using seed crypto, seeddb.bin, looked for next to the dump, in the current folder and in Azahar's or
              Citra's sysdata folder; tools/extract.py's --boot9/--keys/--seeddb/--seed options pass through), or a
              homebrew .3dsx
  <name>      the folder name (data/<name>, generated/<name>, build-<name>/<name>); default: from the dump's file name
  --app       also make an app: tools/bundle_macos.sh on a Mac, tools/bundle_linux.sh elsewhere
  --launcher  (also) build one program with a game list holding every game in data/ that is recompiled
              (build-all/recomp, cmake -DGAMES=all), and with --app an app of it
  --force     extract again even if data/<name> already holds this title

Each step is skipped when its result is already there; a build only recompiles what changed.
"""
import json
import os
import platform
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_OPTS = ('--boot9', '--keys', '--seeddb', '--seed', '--argv0')


def run(cmd, **kw):
    print('$ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, **kw)


def default_name(dump):
    """'000400000010C000 Kirby Triple Deluxe (CTR-P-BALP) (v0.0.0) (E).legit.cia' -> 'kirbytripledeluxe'"""
    base = os.path.splitext(os.path.basename(dump))[0]
    base = re.sub(r'^[0-9A-Fa-f]{16}\s+', '', base)  # a title id in front
    base = re.split(r'[\(\[]', base)[0]
    name = re.sub(r'[^a-z0-9]', '', base.lower())
    return name or 'game'


def program_id(gamedir):
    try:
        return json.load(open(os.path.join(gamedir, 'info.json')))['program_id']
    except (OSError, KeyError, ValueError):
        return None


def build(builddir, defs, target=None):
    if not os.path.exists(os.path.join(ROOT, builddir, 'build.ninja')):
        cc = shutil.which('clang') or 'cc'
        cxx = shutil.which('clang++') or 'c++'
        run(['cmake', '-S', '.', '-B', builddir, '-G', 'Ninja', '-DCMAKE_C_COMPILER=' + cc,
             '-DCMAKE_CXX_COMPILER=' + cxx] + ['-D%s=%s' % kv for kv in defs.items()])
    jobs = ['-j', os.environ['NINJA_JOBS']] if os.environ.get('NINJA_JOBS') else []
    run(['ninja', '-C', builddir] + jobs + ([target] if target else []))


def bundle(args, env=None):
    script = 'tools/bundle_macos.sh' if platform.system() == 'Darwin' else 'tools/bundle_linux.sh'
    run(['sh', script] + args, env=env)


def main():
    argv = sys.argv[1:]
    flags = {a for a in argv if a in ('--app', '--launcher', '--force')}
    passthrough = []
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in EXTRACT_OPTS and i + 1 < len(argv):
            passthrough += [a, argv[i + 1]]
            i += 2
            continue
        if a == '--jobs' and i + 1 < len(argv):
            os.environ['NINJA_JOBS'] = argv[i + 1]
            i += 2
            continue
        if a not in flags:
            rest.append(a)
        i += 1
    if not rest and '--launcher' not in flags:
        print(__doc__)
        sys.exit(1)
    if rest:
        dump = os.path.abspath(rest[0])
        name = rest[1] if len(rest) > 1 else default_name(dump)
        gamedir = os.path.join('data', name)
        # 1. extract (unless data/<name> already holds a game: the same title is kept, another one is an error)
        if '--force' in flags or not program_id(os.path.join(ROOT, gamedir)):
            run([sys.executable, 'tools/extract.py', dump, gamedir, '--no-unpack'] + passthrough)
        else:
            print('%s already extracted (title %s; --force extracts again)' % (gamedir, program_id(os.path.join(ROOT, gamedir))))
        # 2. recompile (regen.sh folds in code earlier runs found missing; files that did not change stay as they are)
        run(['sh', 'tools/regen.sh', gamedir, name])
        # 3. build
        builddir = 'build-' + name
        build(builddir, {'GAME': name})
        print('\nplay: ./%s/%s %s' % (builddir, name, gamedir))
        # 4. app
        if '--app' in flags:
            env = dict(os.environ, BUILD=builddir)
            bundle([name, gamedir], env=env)
    if '--launcher' in flags:
        games = sorted(g for g in os.listdir(os.path.join(ROOT, 'generated'))
                       if os.path.exists(os.path.join(ROOT, 'generated', g, 'table.c'))
                       and program_id(os.path.join(ROOT, 'data', g)))
        if not games:
            sys.exit('no recompiled game with its files in data/')
        build('build-all', {'GAMES': 'all'})
        print('\nplay: ./build-all/recomp  (games: %s)' % ', '.join(games))
        if '--app' in flags:
            env = dict(os.environ, BUILD='build-all')
            bundle(['--games'] + games, env=env)


if __name__ == '__main__':
    main()
