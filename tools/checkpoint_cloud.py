#!/usr/bin/env python3
"""Upload Checkpoint's save backups to the cloud saves' bucket, as one more device's saves.

  python3 tools/checkpoint_cloud.py <Checkpoint saves folder> <file with the s3:// line> [--upload] [--device NAME]
                                    [--as <unique id>=<name>]...

Checkpoint (a 3DS homebrew app) backs saves up as <saves>/0x<unique id> <title>/<YYYYMMDD-HHMMSS>/<files>. Each
game's newest backup becomes <title id>/<device id>.3dssave, as runtime/cloud.mm writes it: savedata/ as a tar
(no times, sorted names), the device's name, when it was saved (the backup's name, local time), which game files
(a hash of data/<name>/code.bin, for the games here with that title id; none otherwise, so no app uses it) and the
tar's SHA-256 as metadata. Without --upload it only lists what the bucket has and what it would upload.

--as 0x01BFB=metroid also uploads that game's save as data/metroid's (another region's version of the game: its
title id and game files), for an app to use it. Whether a save works in another version is up to the game.

A game whose save here has an extdata/ folder keeps it in the upload (Checkpoint's save backups have none, and an
app taking this save replaces both folders).
"""
import datetime
import hashlib
import hmac
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

SAVE_ROOT = os.path.expanduser('~/Library/Application Support/3dsrecomp')
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
TAR_BLOCK = 512
USTAR_NAME, USTAR_PREFIX = 100, 155


def parse_conn(line):
    """s3://<key ID>:<secret>@<endpoint>/<bucket>[/<folder>], as cloud.mm's parse_conn"""
    line = line.strip()
    scheme = 'https'
    if line.startswith('s3://'):
        line = line[5:]
    elif line.startswith('s3+http://'):
        line, scheme = line[10:], 'http'
    else:
        sys.exit('the connection should start with s3://')
    at, colon = line.rfind('@'), line.find(':')
    where = line[at + 1:]
    host, _, path = where.partition('/')
    bucket, _, prefix = path.strip('/').partition('/')
    region = 'us-east-1'
    parts = host.split('.')
    if host.startswith('s3.') and len(parts) > 3:  # s3.<region>.backblazeb2.com
        region = parts[1]
    return dict(scheme=scheme, host=host, region=region, bucket=bucket, prefix=prefix,
                key_id=urllib.parse.unquote(line[:colon]), secret=urllib.parse.unquote(line[colon + 1:at]))


def request(c, method, key='', query=None, body=b'', headers=None):
    """A signed (AWS Signature V4) request: (status, headers, body)"""
    now = datetime.datetime.now(datetime.timezone.utc)
    amz, day = now.strftime('%Y%m%dT%H%M%SZ'), now.strftime('%Y%m%d')
    path = '/' + c['bucket'] + ('/' + urllib.parse.quote(key, safe='/~') if key else '')
    qs = '&'.join(f'{urllib.parse.quote(k, safe="~")}={urllib.parse.quote(v, safe="~")}'
                  for k, v in sorted((query or {}).items()))
    h = {'host': c['host'], 'x-amz-date': amz, 'x-amz-content-sha256': hashlib.sha256(body).hexdigest()}
    h.update({k.lower(): v for k, v in (headers or {}).items()})
    signed = ';'.join(sorted(h))
    canon = '\n'.join([method, path, qs, ''.join(f'{k}:{h[k].strip()}\n' for k in sorted(h)), signed,
                       h['x-amz-content-sha256']])
    scope = f'{day}/{c["region"]}/s3/aws4_request'
    to_sign = '\n'.join(['AWS4-HMAC-SHA256', amz, scope, hashlib.sha256(canon.encode()).hexdigest()])
    k = ('AWS4' + c['secret']).encode()
    for part in (day, c['region'], 's3', 'aws4_request'):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    sig = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    h['authorization'] = f'AWS4-HMAC-SHA256 Credential={c["key_id"]}/{scope}, SignedHeaders={signed}, Signature={sig}'
    url = f'{c["scheme"]}://{c["host"]}{path}' + ('?' + qs if qs else '')
    req = urllib.request.Request(url, data=body if method == 'PUT' else None, method=method,
                                 headers={k: v for k, v in h.items() if k != 'host'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()


def slots(c, tid):
    """The game's .3dssave objects: [(key, device, saved, build)]"""
    prefix = (c['prefix'] + '/' if c['prefix'] else '') + tid + '/'
    st, _, body = request(c, 'GET', query={'list-type': '2', 'prefix': prefix})
    if st != 200:
        sys.exit(f'listing {prefix} failed: HTTP {st} {body[:300]!r}')
    ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    out = []
    for item in ET.fromstring(body).findall('s3:Contents', ns):
        key = item.find('s3:Key', ns).text
        if not key.endswith('.3dssave'):
            continue
        st, h, _ = request(c, 'HEAD', key)
        out.append((key, urllib.parse.unquote(h.get('x-amz-meta-nx-device', '?')),
                    int(h.get('x-amz-meta-nx-saved', 0)), h.get('x-amz-meta-nx-build', '')))
    return out


def tar_add(t, name, data, is_dir):
    """One ustar entry, byte for byte as cloud.mm's tar_add"""
    n, prefix = name + '/' if is_dir else name, ''
    if len(n) > USTAR_NAME:
        cut = n.rfind('/', 0, len(n) - 1)
        while cut != -1 and (cut > USTAR_PREFIX or len(n) - cut - 1 > USTAR_NAME):
            cut = n.rfind('/', 0, cut) if cut else -1
        if cut != -1:
            prefix, n = n[:cut], n[cut + 1:]
    h = bytearray(TAR_BLOCK)
    h[0:len(n[:USTAR_NAME])] = n[:USTAR_NAME].encode()
    h[100:108] = b'%07o\0' % (0o755 if is_dir else 0o644)
    h[108:116] = b'%07o\0' % 0
    h[116:124] = b'%07o\0' % 0
    h[124:136] = b'%011o\0' % (0 if is_dir else len(data))
    h[136:148] = b'%011o\0' % 0
    h[156] = ord('5' if is_dir else '0')
    h[257:263] = b'ustar\0'
    h[263:265] = b'00'
    h[345:345 + len(prefix)] = prefix.encode()
    h[148:156] = b' ' * 8
    h[148:155] = b'%06o\0' % sum(h)
    t += h
    if not is_dir:
        t += data + b'\0' * ((TAR_BLOCK - len(data) % TAR_BLOCK) % TAR_BLOCK)


def add_tree(t, root, rel):
    tar_add(t, rel, b'', True)
    for n in sorted(os.listdir(root)):
        p = os.path.join(root, n)
        if os.path.isdir(p):
            add_tree(t, p, rel + '/' + n)
        elif os.path.isfile(p):
            tar_add(t, rel + '/' + n, open(p, 'rb').read(), False)


def builds():
    """title id -> the hash of its code.bin, for the games in data/"""
    out = {}
    for g in sorted(os.listdir(DATA)):
        info = os.path.join(DATA, g, 'info.json')
        if os.path.isfile(info):
            tid = json.load(open(info))['program_id'].lower()
            out[tid] = (g, hashlib.sha256(open(os.path.join(DATA, g, 'code.bin'), 'rb').read()).hexdigest()[:16])
    return out


def upload_one(c, tid, game, build, backup, saved, device, device_id, upload, fmt):
    """The backup as <title id>/<device id>.3dssave (game: the name in data/ with that title id, if any)"""
    t = bytearray()
    add_tree(t, backup, 'savedata')
    ext = os.path.join(SAVE_ROOT, tid, 'extdata')
    if game and os.path.isdir(ext):
        add_tree(t, ext, 'extdata')
    t += b'\0' * (2 * TAR_BLOCK)
    tar = bytes(t)
    sha = hashlib.sha256(tar).hexdigest()
    key = (c['prefix'] + '/' if c['prefix'] else '') + f'{tid}/{device_id}.3dssave'
    print(f'  as {tid}, {len(tar) // 1024} KB'
          + (f', for {game}' + (' (+ the extdata here)' if os.path.isdir(ext) else '') if game else ''))
    for k, dev, when, b in slots(c, tid):
        same = ' same game files' if build and b == build else ''
        print(f'    in the bucket: {dev} {fmt(when)}{same}' + (' (this one)' if k == key else ''))
    if upload:
        meta = {'x-amz-meta-nx-device': urllib.parse.quote(device), 'x-amz-meta-nx-saved': str(saved),
                'x-amz-meta-nx-build': build, 'x-amz-meta-nx-hash': sha}
        st, _, body = request(c, 'PUT', key, body=tar, headers=meta)
        print(f'    uploaded {key}' if st == 200 else f'    UPLOAD FAILED: HTTP {st} {body[:300]!r}')


def main():
    args = sys.argv[1:]
    upload = '--upload' in args
    device = '3DS'
    if '--device' in args:
        device = args[args.index('--device') + 1]
        del args[args.index('--device'):args.index('--device') + 2]
    aliases = {}  # title id -> the name in data/ whose save it also is
    while '--as' in args:
        i = args.index('--as')
        uid, _, name = args[i + 1].partition('=')
        aliases['00040000' + f'{int(uid, 16):06x}' + '00'] = name
        del args[i:i + 2]
    args = [a for a in args if a != '--upload']
    if len(args) != 2:
        sys.exit(__doc__)
    saves, conn_file = args
    c = parse_conn(open(conn_file).read())
    device_id = device.lower().replace(' ', '-') + '-checkpoint'
    known = builds()
    fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M') if t else '-'
    for folder in sorted(os.listdir(saves)):
        if not folder.startswith('0x'):
            continue
        uid, _, title = folder[2:].partition(' ')
        tid = '00040000' + f'{int(uid, 16):06x}' + '00'  # (unique id 0x0053F: title id 0004000000053f00)
        src = os.path.join(saves, folder)
        backups = sorted(b for b in os.listdir(src) if os.path.isdir(os.path.join(src, b)))
        if not backups:
            continue
        newest = backups[-1]
        saved = int(datetime.datetime.strptime(newest, '%Y%m%d-%H%M%S').timestamp())
        dests = [(tid, *known.get(tid, (None, '')))]  # (title id, game, its build)
        if tid in aliases:
            other = [t for t, (g, _) in known.items() if g == aliases[tid]]
            if not other:
                sys.exit(f'--as: no data/{aliases[tid]}')
            dests.append((other[0], *known[other[0]]))
        print(f'{tid} {" ".join(title.split())}: backup {newest}')
        for dtid, game, build in dests:
            upload_one(c, dtid, game, build, os.path.join(src, newest), saved, device, device_id, upload, fmt)


if __name__ == '__main__':
    main()
