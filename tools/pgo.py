#!/usr/bin/env python3
"""Profile-guided optimisation of a game's recompiled code: its regression tests' runs tell the compiler which code
is hot.

  tools/pgo.py <name>... [--checkpoints <folder of Checkpoint backups>] [--tests <test>,...]

For each game (data/<name>, built in the build folder configured with GAME=<name>):

  1. an instrumented build of the same code in build-pgo-<name> (cmake -DGEN_PROFILE=generate)
  2. the game's tests (tools/gametest.py, deterministic runs) played with it, each run writing its counts
  3. the counts merged into pgo/<name>.profdata
  4. the game's own build reconfigured with -DGEN_PROFILE=pgo/<name>.profdata and rebuilt

The compiler then lays out, inlines and allocates registers for the code the runs went through; the code they did
not reach is compiled as before (CMakeLists.txt: GEN_PROFILE). A profile belongs to the generated code it was made
with: after tools/regen.sh changes a function, run this again (until then the functions that changed are compiled
without one). A profile holds counts and function names, nothing of the game; pgo/ is not part of the repository.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, **kw):
    print('$ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, **kw)


def cache_value(build, key):
    m = re.search(r'^%s:[A-Z]+=(.*)$' % re.escape(key), open(os.path.join(build, 'CMakeCache.txt')).read(), re.M)
    return m.group(1).strip() if m else None


def game_build(name):
    """The game's own build folder (configured with GAME=<name>, not an instrumented one)."""
    for cache in sorted(glob.glob(os.path.join(ROOT, 'build*', 'CMakeCache.txt'))):
        d = os.path.dirname(cache)
        if os.path.basename(d).startswith('build-pgo-'):
            continue
        if cache_value(d, 'GAME') == name and not cache_value(d, 'GAMES'):
            return d
    sys.exit('no build of %s (a build folder configured with -DGAME=%s): tools/port.py makes one' % (name, name))


def profdata_tool(cc):
    """llvm-profdata of the compiler's own version (a profile's format follows the compiler's)."""
    for cand in (os.path.join(os.path.dirname(os.path.realpath(cc)), 'llvm-profdata'), shutil.which('llvm-profdata')):
        if cand and os.access(cand, os.X_OK):
            return [cand]
    if sys.platform == 'darwin':
        return ['xcrun', 'llvm-profdata']
    sys.exit('no llvm-profdata (it comes with LLVM: the llvm package)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('names', nargs='+', help='games (data/<name>)')
    ap.add_argument('--checkpoints', help='your Checkpoint backups, for the tests that start from a save')
    ap.add_argument('--tests', help='the tests to play (default: all of the game\'s)')
    args = ap.parse_args()
    for name in args.names:
        tid = json.load(open(os.path.join(ROOT, 'data', name, 'info.json')))['program_id']
        build = game_build(name)
        cc = cache_value(build, 'CMAKE_C_COMPILER') or 'clang'
        cxx = cache_value(build, 'CMAKE_CXX_COMPILER') or 'clang++'
        inst = os.path.join(ROOT, 'build-pgo-' + name)
        run(['cmake', '-S', ROOT, '-B', inst, '-G', 'Ninja', '-DCMAKE_C_COMPILER=' + cc, '-DCMAKE_CXX_COMPILER=' + cxx,
             '-DGAME=' + name, '-DGEN_PROFILE=generate'], stdout=subprocess.DEVNULL)
        run(['ninja', '-C', inst])
        raw = os.path.join(ROOT, 'pgo', 'raw-' + name)
        shutil.rmtree(raw, ignore_errors=True)
        os.makedirs(raw)
        env = dict(os.environ, LLVM_PROFILE_FILE=os.path.join(raw, '%p-%m.profraw'))
        sel = args.tests.split(',') if args.tests else [tid]
        test = [sys.executable, 'tools/gametest.py', '--builds', os.path.basename(inst)] + sel
        if args.checkpoints:
            test += ['--checkpoints', args.checkpoints]
        r = subprocess.run(test, cwd=ROOT, env=env)
        if r.returncode:
            print('warning: some of the tests failed; their runs still count', file=sys.stderr)
        profs = glob.glob(os.path.join(raw, '*.profraw'))
        if not profs:
            sys.exit('no profile was written (did a test run?)')
        out = os.path.join(ROOT, 'pgo', name + '.profdata')
        run(profdata_tool(cc) + ['merge', '-o', out] + profs)
        shutil.rmtree(raw)
        run(['cmake', '-B', build, '-DGEN_PROFILE=' + out], stdout=subprocess.DEVNULL)
        run(['ninja', '-C', build])
        print('%s: optimised with %s (%d runs)' % (name, os.path.relpath(out, ROOT), len(profs)))


if __name__ == '__main__':
    main()
