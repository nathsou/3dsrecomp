#!/usr/bin/env python3
"""Regression tests for recompiled games: scripted headless runs checked against expectations.

  tools/gametest.py [options] [title id or test name ...]

Each test is a TOML file in tests/games/<title id>/ (see tests/README.md for the format). A test starts the game
headless (SDL's offscreen video and dummy audio) with scripted input, optionally from a save of your own console,
and checks:

  - the run ends normally, with no fatal error and no code the recompiler missed (lazily recompiled code);
  - every "unimplemented command" the services log is one the test lists (`known_unimplemented`), so a game that
    starts using a new command is noticed;
  - at its checkpoints: screens that are not black, the profile's values (RECOMP_PROFILE_TRACE), and, when a
    reference picture exists locally, a picture close to it;
  - optionally, the game's frame rate over a stretch of frames (only measured when tests run one at a time,
    `--jobs 1`: other processes skew it).

Nothing from the games is kept in the repository: the game files come from data/, the builds from build*/, saves
from your Checkpoint backups (--checkpoints, or $RECOMP_TEST_CHECKPOINTS), reference pictures from --refs
(default ~/.local/share/3dsrecomp/test-refs, or $RECOMP_TEST_REFS; --update-refs writes them). A test whose game,
build or save is missing here is skipped. Each run's log and pictures go to shots/tests/<title id>-<test>/, with a
contact sheet (sheet.png) when Pillow is installed.
"""
import argparse
import concurrent.futures
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from PIL import Image
except ImportError:  # pictures are then only checked for size and brightness by hand-rolled PPM reading
    Image = None


# ---------------------------------------------------------------------------------------------------- discovery
def find_games():
    """data/<name> folders by title id (lowercase hex)."""
    games = {}
    for info in glob.glob(os.path.join(ROOT, 'data', '*', 'info.json')):
        try:
            tid = json.load(open(info))['program_id'].lower()
        except (OSError, KeyError, ValueError):
            continue
        games.setdefault(tid, os.path.dirname(info))
    return games


def find_executable(name):
    """The build of data/<name>: a build directory configured with GAME=<name> holding the executable."""
    for cache in sorted(glob.glob(os.path.join(ROOT, 'build*', 'CMakeCache.txt'))):
        d = os.path.dirname(cache)
        try:
            text = open(cache).read()
        except OSError:
            continue
        m = re.search(r'^GAME:STRING=(.*)$', text, re.M)
        if m and m.group(1).strip() == name and os.access(os.path.join(d, name), os.X_OK):
            return os.path.join(d, name)
    return None


def find_checkpoint_save(root, tid, uid=None):
    """The newest Checkpoint backup of a title: <root>/0x<unique id> <title>/<date>/ (uid: another title's unique
    id, for a game that keeps the save of an earlier edition)."""
    if not root or not os.path.isdir(root):
        return None
    uid = uid or '0x%05X' % ((int(tid, 16) >> 8) & 0xFFFFF)
    for d in sorted(os.listdir(root)):
        if d.upper().startswith(uid.upper() + ' ') or d.upper() == uid.upper():
            dates = sorted(x for x in os.listdir(os.path.join(root, d)) if os.path.isdir(os.path.join(root, d, x)))
            if dates:
                return os.path.join(root, d, dates[-1])
    return None


# ---------------------------------------------------------------------------------------------------- input
def expand_input(items):
    """RECOMP_INPUT from a list of events: plain `frame:BUTTONS:duration` (or any RECOMP_INPUT event), or
    `BUTTONS:duration every N from F to T` for a repeated press."""
    if isinstance(items, str):
        items = [items]
    out = []
    for it in items:
        for part in [p.strip() for p in it.split(',') if p.strip()]:
            m = re.fullmatch(r'(\S+?):(\d+)\s+every\s+(\d+)\s+from\s+(\d+)\s+to\s+(\d+)', part)
            if m:
                ev, dur, every, f0, f1 = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
                out += ['%d:%s:%d' % (f, ev, dur) for f in range(f0, f1, every)]
            else:
                out.append(part)
    return ','.join(out)


# ---------------------------------------------------------------------------------------------------- pictures
def read_ppm(path):
    if Image:
        return Image.open(path).convert('RGB')
    with open(path, 'rb') as f:
        data = f.read()
    m = re.match(rb'P6\s+(\d+)\s+(\d+)\s+(\d+)\s', data)
    w, h = int(m.group(1)), int(m.group(2))
    return (w, h, data[m.end():])


def luma_stats(img):
    """Mean and standard deviation of the luminance (0-255)."""
    if Image:
        px = img.convert('L').resize((80, 48)).tobytes()
    else:
        w, h, raw = img
        px = bytes((raw[i] * 3 + raw[i + 1] * 6 + raw[i + 2]) // 10 for i in range(0, len(raw), 3 * 37))
    n = len(px)
    mean = sum(px) / n
    sd = math.sqrt(sum((p - mean) ** 2 for p in px) / n)
    return mean, sd


def difference(a, b):
    """Mean absolute difference (0-255) of two pictures shrunk to 40x24: robust to small timing differences."""
    if not Image:
        return None
    x = a.convert('RGB').resize((40, 24)).tobytes()
    y = b.convert('RGB').resize((40, 24)).tobytes()
    return sum(abs(p - q) for p, q in zip(x, y)) / len(x)


def contact_sheet(run_dir, frames):
    if not Image:
        return
    cells = []
    for f in frames:
        t = os.path.join(run_dir, 'shot_top_%06d.ppm' % f)
        b = os.path.join(run_dir, 'shot_bottom_%06d.ppm' % f)
        if not os.path.exists(t):
            continue
        im = Image.new('RGB', (400, 480), (40, 0, 40))
        im.paste(Image.open(t).convert('RGB').resize((400, 240)), (0, 0))
        if os.path.exists(b):
            im.paste(Image.open(b).convert('RGB').resize((320, 240)), (40, 240))
        cells.append(im)
    if not cells:
        return
    cols = min(6, len(cells))
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * 400, rows * 480))
    for i, c in enumerate(cells):
        sheet.paste(c, ((i % cols) * 400, (i // cols) * 480))
    sheet.save(os.path.join(run_dir, 'sheet.png'))


# ---------------------------------------------------------------------------------------------------- one test
class Result:
    def __init__(self, test):
        self.test = test
        self.status = 'pass'  # pass, fail, skip
        self.notes = []
        self.seconds = 0.0

    def fail(self, msg):
        self.status = 'fail'
        self.notes.append(msg)

    def note(self, msg):
        self.notes.append(msg)


def load_tests(selectors):
    tests = []
    for path in sorted(glob.glob(os.path.join(ROOT, 'tests', 'games', '*', '*.toml'))):
        tid = os.path.basename(os.path.dirname(path)).lower()
        with open(path, 'rb') as f:
            spec = tomllib.load(f)
        spec['tid'] = tid
        spec['name'] = spec.get('name', os.path.splitext(os.path.basename(path))[0])
        spec['path'] = path
        spec['id'] = '%s-%s' % (tid, spec['name'])
        if selectors and not any(s.lower() in (tid, spec['name'], spec['id'], spec.get('game', '').lower())
                                 for s in selectors):
            continue
        tests.append(spec)
    return tests


def run_test(spec, args, games):
    r = Result(spec)
    tid = spec['tid']
    gamedir = games.get(tid)
    if not gamedir:
        r.status = 'skip'
        r.note('no data/ folder with title %s' % tid)
        return r
    exe = find_executable(os.path.basename(gamedir))
    if not exe:
        r.status = 'skip'
        r.note('no build of %s (a build directory configured with GAME=%s)' % (gamedir, os.path.basename(gamedir)))
        return r
    run_dir = os.path.join(ROOT, 'shots', 'tests', spec['id'])
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    save_kind = spec.get('save')
    if save_kind == 'checkpoint':
        src = find_checkpoint_save(args.checkpoints, tid, spec.get('checkpoint_id'))
        if not src:
            r.status = 'skip'
            r.note('no Checkpoint backup of %s under %s' % (tid, args.checkpoints or '(no --checkpoints)'))
            return r
        dst = os.path.join(run_dir, 'save', tid, 'savedata')
        shutil.copytree(src, dst)
        for d in spec.get('save_dirs', []):  # Checkpoint leaves out empty folders
            os.makedirs(os.path.join(dst, d), exist_ok=True)
        r.note('save: %s' % src)
    checkpoints = spec.get('checkpoint', [])
    frames = int(spec['frames'])
    shot_frames = sorted({int(c['frame']) for c in checkpoints})
    env = dict(os.environ)
    env.update({'SDL_VIDEO_DRIVER': 'offscreen', 'SDL_AUDIO_DRIVER': 'dummy', 'RECOMP_FRAMES': str(frames)})
    for k in list(env):
        if k.startswith(('RECOMP_', 'GPU_')) and k not in ('RECOMP_FRAMES',):
            del env[k]  # (the caller's debugging variables would change the run)
    if shot_frames:
        step = 0
        for f in shot_frames:
            step = math.gcd(step, f)
        env['RECOMP_SHOT'] = str(step)
        env['RECOMP_SHOT_FROM'] = str(shot_frames[0])
    if spec.get('input'):
        env['RECOMP_INPUT'] = expand_input(spec['input'])
    if 'swkbd' in spec:
        env['RECOMP_SWKBD_TEXT'] = spec['swkbd']
    if any('values' in c for c in checkpoints):
        env['RECOMP_PROFILE_TRACE'] = '10'
    perf = spec.get('fps')
    if perf:
        env['RECOMP_STATS'] = '1'
    for k, v in spec.get('env', {}).items():
        env[k] = str(v)
    timeout = spec.get('timeout', frames / 60 * 3 + 120)
    t0 = time.time()
    with open(os.path.join(run_dir, 'run.log'), 'w') as log:
        try:
            rc = subprocess.run([exe, gamedir], cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = None
    r.seconds = time.time() - t0
    text = open(os.path.join(run_dir, 'run.log'), errors='replace').read()
    if rc is None:
        r.fail('timed out after %d s' % timeout)
    elif rc != 0:
        r.fail('exit code %d' % rc)
    for line in text.splitlines():
        if re.search(r'\bfatal\b|FATAL|Assertion|Segmentation', line):
            r.fail('log: ' + line.strip()[:160])
            break
    lazy = sorted(set(re.findall(r'\[lazy\] recompiling missing code at ([0-9a-f]+)', text)))
    if lazy and not spec.get('allow_lazy'):
        r.fail('code the recompiler missed: %s (tools/regen.sh adds it from save/missing_functions.txt)'
               % ' '.join(lazy[:8]))
    seen = set('%s %s' % m for m in re.findall(r'\[([^\]]+)\] unimplemented command (0x[0-9a-f]{8})', text))
    known = set(spec.get('known_unimplemented', []))
    new = sorted(seen - known)
    if new:
        r.fail('new unimplemented commands: %s' % ', '.join(new))
    gone = sorted(known - seen)
    if gone and args.verbose:
        r.note('listed but not seen this run: %s' % ', '.join(gone))
    # checkpoints
    trace = []
    for m in re.finditer(r'^\s*(\d+) \[profile\] focus (\S+) hud (\d+):(.*)$', text, re.M):
        vals = dict(re.findall(r'(\w+)=(-?\d+)', m.group(4)))
        trace.append((int(m.group(1)), m.group(2), vals))
    refs = os.path.join(args.refs, spec['id'])
    for c in checkpoints:
        f = int(c['frame'])
        for screen in c.get('screens', []):
            p = os.path.join(run_dir, 'shot_%s_%06d.ppm' % (screen, f))
            if not os.path.exists(p):
                r.fail('frame %d: no %s screen picture (the run ended early?)' % (f, screen))
                continue
            mean, sd = luma_stats(read_ppm(p))
            if mean < 6 or sd < 2:
                r.fail('frame %d: the %s screen is blank (mean %.0f, deviation %.1f)' % (f, screen, mean, sd))
        if 'values' in c:
            got = [t for t in trace if t[0] <= f]
            if not got:
                r.fail('frame %d: no profile trace (does the game have a profile?)' % f)
            else:
                _, focus, vals = got[-1]
                for k, want in c['values'].items():
                    if k == 'focus':
                        if focus != want:
                            r.fail('frame %d: focus %s, expected %s' % (f, focus, want))
                    elif str(vals.get(k)) != str(want):
                        r.fail('frame %d: %s = %s, expected %s' % (f, k, vals.get(k, '-'), want))
        if c.get('ref', True) and Image:
            for screen in ('top', 'bottom'):
                p = os.path.join(run_dir, 'shot_%s_%06d.ppm' % (screen, f))
                ref = os.path.join(refs, '%06d_%s.png' % (f, screen))
                if not os.path.exists(p):
                    continue
                if args.update_refs:
                    os.makedirs(refs, exist_ok=True)
                    Image.open(p).convert('RGB').save(ref)
                elif os.path.exists(ref):
                    d = difference(Image.open(p), Image.open(ref))
                    limit = c.get('ref_tolerance', spec.get('ref_tolerance', 24))
                    if d > limit:
                        r.fail('frame %d: %s screen differs from its reference (%.1f > %s)' % (f, screen, d, limit))
    if perf:
        windows = [(int(m.group(1)), int(m.group(2))) for m in
                   re.finditer(r'^\s*(\d+) \[stats\] \d+ vblanks/2s, (\d+) game frames', text, re.M)]
        sel = [n / 2 for fr, n in windows if perf['from'] <= fr <= perf['to']]
        if args.jobs != 1:
            r.note('frame rate not measured (tests in parallel; --jobs 1 measures it)')
        elif not sel:
            r.fail('no frame rate statistics between frames %d and %d' % (perf['from'], perf['to']))
        else:
            avg = sum(sel) / len(sel)
            r.note('%.1f fps between frames %d and %d (lowest window %.1f)' % (avg, perf['from'], perf['to'], min(sel)))
            if avg < perf['min']:
                r.fail('%.1f fps between frames %d and %d, expected at least %s' % (avg, perf['from'], perf['to'],
                                                                                 perf['min']))
    contact_sheet(run_dir, shot_frames)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('select', nargs='*', help='title ids, test names or <title id>-<test> (default: all)')
    ap.add_argument('--checkpoints', default=os.environ.get('RECOMP_TEST_CHECKPOINTS'),
                    help='the folder holding your Checkpoint backups (<0xID title>/<date>/)')
    ap.add_argument('--refs', default=os.environ.get('RECOMP_TEST_REFS',
                                                     os.path.expanduser('~/.local/share/3dsrecomp/test-refs')),
                    help='reference pictures (kept out of the repository)')
    ap.add_argument('--update-refs', action='store_true', help='save this run\'s checkpoint pictures as references')
    ap.add_argument('-j', '--jobs', type=int, default=max(1, min(4, (os.cpu_count() or 2) // 4)),
                    help='tests run at once (frame rates are only checked with 1)')
    ap.add_argument('-l', '--list', action='store_true', help='list the tests and whether they can run here')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()
    tests = load_tests(args.select)
    games = find_games()
    if not tests:
        sys.exit('no tests match')
    if args.list:
        for t in tests:
            gd = games.get(t['tid'])
            exe = find_executable(os.path.basename(gd)) if gd else None
            save = t.get('save')
            ok = gd and exe and (save != 'checkpoint' or find_checkpoint_save(args.checkpoints, t['tid'], t.get('checkpoint_id')))
            print('%-34s %-5s %s' % (t['id'], 'ready' if ok else 'skip', t.get('description', '')))
        return
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(run_test, t, args, games): t for t in tests}
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            results.append(r)
            print('%-5s %-34s %5.0fs  %s' % (r.status.upper(), r.test['id'], r.seconds,
                                              ('; '.join(r.notes))[:300]), flush=True)
    counts = {s: sum(1 for r in results if r.status == s) for s in ('pass', 'fail', 'skip')}
    print('\n%d passed, %d failed, %d skipped' % (counts['pass'], counts['fail'], counts['skip']))
    sys.exit(1 if counts['fail'] else 0)


if __name__ == '__main__':
    main()
