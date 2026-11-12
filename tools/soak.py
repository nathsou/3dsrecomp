#!/usr/bin/env python3
"""Soak runs: each game played for a long time with random input, to find what breaks before a player does.

  tools/soak.py [--frames N] [--seeds 1,2,...] [--jobs N] [title id or data name ...]

A run is deterministic (RECOMP_DETERMINISTIC): the input comes from a seed, so a run that finds something can be
played again exactly (the input is saved as input.txt next to its log; RECOMP_INPUT="$(cat input.txt)" replays
it). The input presses buttons (A often during the first minute, to get through the menus), holds the circle pad
in a direction, sometimes while pressing a button, taps the touch screen and sometimes presses START. A run starts
from your newest Checkpoint backup of the game when there is one (--checkpoints, or $RECOMP_TEST_CHECKPOINTS), else
from a new save.

What is reported, grouped by kind (numbers and addresses left out): fatal errors, code the recompiler missed,
unimplemented service commands, unsupported GPU features, games that stop submitting GPU work, and other lines
that say something failed. Logs and pictures go to shots/soak/<name>-<seed>/.
"""
import argparse
import collections
import concurrent.futures
import json
import os
import random
import re
import shutil
import subprocess
import sys

import gametest

ROOT = gametest.ROOT
PROBLEMS = re.compile(r'FATAL|\[lazy\]|unimplemented|not supported|unsupported|unknown|\[diag\]|ignored|\bbad\b|'
                      r'failed|invalid|not implemented|KEY MISMATCH|rewritten by the game', re.I)
STICK = ['CUP', 'CDOWN', 'CLEFT', 'CRIGHT', 'CUP+A', 'CRIGHT+B', 'CLEFT+Y']
BUTTONS = ['A', 'A', 'B', 'X', 'Y', 'L', 'R', 'DUP', 'DDOWN', 'DLEFT', 'DRIGHT']
MENU_FRAMES = 3000  # (the first minute: mostly A, to get through the menus)


def random_input(rng, frames):
    """RECOMP_INPUT events: one every 15 to 60 frames."""
    ev = []
    f = 300
    while f < frames:
        r = rng.random()
        if f < MENU_FRAMES and r < 0.5:
            ev.append('%d:A:6' % f)
        elif r < 0.30:
            ev.append('%d:%s:6' % (f, rng.choice(BUTTONS)))
        elif r < 0.55:
            ev.append('%d:%s:%d' % (f, rng.choice(STICK), rng.randint(30, 200)))
        elif r < 0.75:
            ev.append('%d:T@%d/%d:%d' % (f, rng.randint(10, 310), rng.randint(10, 230), rng.randint(4, 30)))
        elif r < 0.78:
            ev.append('%d:START:6' % f)
        f += rng.randint(15, 60)
    return ','.join(ev)


def kind(line):
    """A log line without its frame number, addresses and numbers: what kind of problem it is."""
    line = re.sub(r'^\s*\d+ ', '', line)
    line = re.sub(r'\b(0x)?[0-9a-fA-F]{6,16}\b', 'X', line)
    line = re.sub(r'\b\d+\b', 'N', line)
    return line.strip()[:150]


def soak(gamedir, seed, args):
    name = os.path.basename(gamedir)
    tid = json.load(open(os.path.join(gamedir, 'info.json')))['program_id'].lower()
    exe = gametest.find_executable(name)
    if not exe:
        return name, seed, 'no build of %s' % name, collections.Counter()
    d = os.path.join(ROOT, 'shots', 'soak', '%s-%d' % (name, seed))
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    src = gametest.find_checkpoint_save(args.checkpoints, tid)
    if src:
        shutil.copytree(src, os.path.join(d, 'save', tid, 'savedata'))
    # (the seed of a game's input: stable across runs, so a run can be played again)
    script = random_input(random.Random(seed * 1000 + sum(name.encode()) % 1000), args.frames)
    open(os.path.join(d, 'input.txt'), 'w').write(script)
    env = dict(os.environ)
    for k in list(env):
        if k.startswith(('RECOMP_', 'GPU_')):
            del env[k]
    env.update({'SDL_VIDEO_DRIVER': 'dummy' if sys.platform == 'darwin' else 'offscreen', 'SDL_AUDIO_DRIVER': 'dummy',
                'RECOMP_DETERMINISTIC': '1', 'RECOMP_FRAMES': str(args.frames), 'RECOMP_INPUT': script,
                'RECOMP_SHOT': str(args.frames // 4), 'RECOMP_SWKBD_TEXT': 'Ann'})
    try:
        rc = subprocess.run([exe, gamedir], cwd=d, env=env, stdout=open(os.path.join(d, 'run.log'), 'w'),
                            stderr=subprocess.STDOUT, timeout=args.frames / 60 * 2 + 300).returncode
        status = 'ok' if rc == 0 else 'exit code %d' % rc
    except subprocess.TimeoutExpired:
        status = 'timed out'
    text = open(os.path.join(d, 'run.log'), errors='replace').read()
    frames = [int(m) for m in re.findall(r'^\s*(\d+) ', text, re.M)]
    status += ', %s save, reached frame %d' % ('Checkpoint' if src else 'new', max(frames or [0]))
    return name, seed, status, collections.Counter(kind(l) for l in text.splitlines() if PROBLEMS.search(l))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('select', nargs='*', help='title ids or data/ folder names (default: every game with a build)')
    ap.add_argument('--frames', type=int, default=30000, help='frames per run (default 30000: about 8 minutes)')
    ap.add_argument('--seeds', default='1', help='comma-separated seeds: a run per seed and game')
    ap.add_argument('--checkpoints', default=os.environ.get('RECOMP_TEST_CHECKPOINTS'),
                    help='the folder holding your Checkpoint backups')
    ap.add_argument('-j', '--jobs', type=int, default=max(1, min(4, (os.cpu_count() or 2) // 4)))
    args = ap.parse_args()
    games = sorted(set(gametest.find_games().values()))
    if args.select:
        games = [g for g in games if os.path.basename(g) in args.select or
                 json.load(open(os.path.join(g, 'info.json')))['program_id'].lower() in args.select]
    runs = [(g, int(s)) for g in games for s in args.seeds.split(',')]
    found = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for name, seed, status, problems in pool.map(lambda r: soak(r[0], r[1], args), runs):
            print('== %s, seed %d: %s' % (name, seed, status))
            for k, n in problems.most_common(12):
                print('   %4d  %s' % (n, k))
            found += bool(problems) or not status.startswith('ok')
            sys.stdout.flush()
    print('\n%d of %d runs found something' % (found, len(runs)))
    sys.exit(1 if found else 0)


if __name__ == '__main__':
    main()
