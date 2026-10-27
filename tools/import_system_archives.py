#!/usr/bin/env python3
"""Imports the open-source replacements of 3DS system archives (Mii data, shared font, bad word list, country
list) from an Azahar/Citra source checkout into system_archives/<title id>.romfs, where the runtime's fs looks
for NCCH archives. usage: import_system_archives.py <azahar source dir> [out dir]"""
import os
import re
import sys

ARCHIVES = {
    'mii.app.romfs.h': '0004009b00010202',
    'country_list.app.romfs.h': '0004009b00010402',
    'shared_font.app.romfs.h': '0004009b00014002',
    'bad_word_list.app.romfs.h': '000400db00010302',
}

src = os.path.join(sys.argv[1], 'externals', 'open_source_archives', 'include')
out = sys.argv[2] if len(sys.argv) > 2 else 'system_archives'
os.makedirs(out, exist_ok=True)
for header, tid in ARCHIVES.items():
    path = os.path.join(src, header)
    if not os.path.exists(path):
        print('missing', path)
        continue
    text = open(path).read()
    body = text[text.index('{') + 1:text.rindex('}')]
    data = bytes(int(x, 16) for x in re.findall(r'0x([0-9a-fA-F]{2})', body))
    with open(os.path.join(out, tid + '.romfs'), 'wb') as f:
        f.write(data)
    print('%s: %d bytes' % (tid, len(data)))
