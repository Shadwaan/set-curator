"""
Read what a USB stick's export.pdb (the database your XDJ-1000 MK1 reads) holds for every song: title, file name, comment.
Read-only: it only reads a copy. Used to see which songs on the stick still carry an old comment.

    python stick_tags.py [path to export.pdb]
"""

import re
import struct
import sys

ROWS_TABLE_TRACKS = 0
STR_COMMENT, STR_TITLE, STR_FILENAME = 16, 17, 19          # string slots of a track row (checked against the library below)


def read_string(b: bytes, pos: int) -> str:
    k = b[pos]
    if k & 1:                                                # short ascii: length in the first byte
        n = (k >> 1) - 1
        return b[pos + 1:pos + 1 + n].decode("latin1", "replace")
    n = struct.unpack_from("<H", b, pos + 1)[0] - 4
    if k == 0x40:
        return b[pos + 4:pos + 4 + n].decode("ascii", "replace")
    if k == 0x90:
        return b[pos + 4:pos + 4 + n].decode("utf-16-le", "replace")
    return ""


def table_rows(b: bytes, table_type: int):
    page_len, ntab = struct.unpack_from("<II", b, 4)[0], struct.unpack_from("<I", b, 8)[0]
    npages = len(b) // page_len
    first = last = None
    for t in range(ntab):
        ttype, _, f, l = struct.unpack_from("<IIII", b, 28 + 16 * t)
        if ttype == table_type:
            first, last = f, l
    page, seen = first, set()
    while page is not None and page not in seen and page < npages:
        seen.add(page)
        base = page * page_len
        n_off = (struct.unpack_from("<I", b, base + 0x18)[0] & 0xFFFFFF) & 0x1FFF
        if not b[base + 0x1B] & 0x40:
            for g in range((n_off + 15) // 16):
                gbase = base + page_len - g * 0x24
                flags = struct.unpack_from("<H", b, gbase - 4)[0]
                for i in range(16):
                    if g * 16 + i >= n_off:
                        break
                    if flags & (1 << i):
                        yield base + 0x28 + struct.unpack_from("<H", b, gbase - 6 - 2 * i)[0]
        page = None if page == last else struct.unpack_from("<I", b, base + 12)[0]


def tracks(path: str) -> list[dict]:
    b = open(path, "rb").read()
    out = []
    for r in table_rows(b, ROWS_TABLE_TRACKS):
        offs = struct.unpack_from("<21H", b, r + 0x5E)
        s = lambda k: read_string(b, r + offs[k]) if offs[k] else ""
        out.append({"id": struct.unpack_from("<I", b, r + 0x48)[0], "title": s(STR_TITLE), "comment": s(STR_COMMENT),
                    "file": s(STR_FILENAME), "tempo": struct.unpack_from("<I", b, r + 0x38)[0] / 100})
    return out


if __name__ == "__main__":
    t = tracks(sys.argv[1])
    print(len(t), "tracks")
    for x in t[:8]:
        print(x)


def playlists(path: str) -> tuple[dict, list]:
    """(tree, entries): tree {id: (parent id, is folder, name)}, entries [(playlist id, position, track id)]."""
    b = open(path, "rb").read()
    tree = {}
    for r in table_rows(b, 7):
        parent, _, _, pid, folder = struct.unpack_from("<5I", b, r)
        tree[pid] = (parent, bool(folder), read_string(b, r + 20))
    entries = []
    for r in table_rows(b, 8):
        pos, tid, pid = struct.unpack_from("<3I", b, r)
        entries.append((pid, pos, tid))
    return tree, entries


def track_paths(path: str) -> dict:
    """{track id: file path stored on the stick}"""
    b = open(path, "rb").read()
    out = {}
    for r in table_rows(b, ROWS_TABLE_TRACKS):
        offs = struct.unpack_from("<21H", b, r + 0x5E)
        out[struct.unpack_from("<I", b, r + 0x48)[0]] = read_string(b, r + offs[20]) if offs[20] else ""
    return out
