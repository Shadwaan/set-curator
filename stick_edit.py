"""
Rewrite song comments inside a stick's export.pdb (the database an XDJ-1000 MK1 reads), on a COPY.

A track row keeps its strings after its fixed fields, so a longer comment will not fit in place. For each song that needs
a new comment this builds a new copy of the row (same fields, same strings, the new comment at the end), puts it in the
free space of the page the row is on, and points the page's row table at it. The old row is left behind as unused space.
A song whose page has no room is skipped and reported. Nothing else in the file changes, and the result is re-read to
check that every track, playlist and playlist entry is still there.

    python stick_edit.py <export.pdb copy> <new file> [--apply from library]    # normally called from code
"""

import struct

import stick_tags

PAGE_FREE, PAGE_USED = 0x1C, 0x1E


def encode(text: str) -> bytes:
    if all(ord(c) < 128 for c in text) and len(text) <= 126:
        return bytes([((len(text) + 1) << 1) | 1]) + text.encode("ascii")
    raw = text.encode("utf-16-le")
    return b"\x90" + struct.pack("<H", len(raw) + 4) + b"\x00" + raw


def string_end(b: bytes, pos: int) -> int:
    k = b[pos]
    if k & 1:
        return pos + (k >> 1)
    return pos + struct.unpack_from("<H", b, pos + 1)[0]


def edit(src: str, dst: str, wanted: dict[str, str]) -> dict:
    """wanted: {normalised file name: new comment}. Returns counts."""
    b = bytearray(open(src, "rb").read())
    page_len = struct.unpack_from("<I", b, 4)[0]
    ntab = struct.unpack_from("<I", b, 8)[0]
    first = last = None
    for t in range(ntab):
        ttype, _, f, l = struct.unpack_from("<IIII", b, 28 + 16 * t)
        if ttype == stick_tags.ROWS_TABLE_TRACKS:
            first, last = f, l
    done = skipped_full = untouched = 0
    page, seen = first, set()
    while page is not None and page not in seen:
        seen.add(page)
        base = page * page_len
        n_off = (struct.unpack_from("<I", b, base + 0x18)[0] & 0xFFFFFF) & 0x1FFF
        if not b[base + 0x1B] & 0x40:
            for g in range((n_off + 15) // 16):
                gbase = base + page_len - g * 0x24
                flags = struct.unpack_from("<H", b, gbase - 4)[0]
                for i in range(16):
                    if g * 16 + i >= n_off or not flags & (1 << i):
                        continue
                    slot = gbase - 6 - 2 * i
                    r = base + 0x28 + struct.unpack_from("<H", b, slot)[0]
                    offs = struct.unpack_from("<21H", b, r + 0x5E)
                    name = stick_tags.read_string(bytes(b), r + offs[stick_tags.STR_FILENAME]) if offs[stick_tags.STR_FILENAME] else ""
                    key = name.lower()
                    new = wanted.get(key)
                    if new is None:
                        continue
                    cur = stick_tags.read_string(bytes(b), r + offs[stick_tags.STR_COMMENT]) if offs[stick_tags.STR_COMMENT] else ""
                    if cur == new:
                        untouched += 1
                        continue
                    end = max(string_end(bytes(b), r + o) for o in offs if o) - r
                    row = bytearray(b[r:r + end])
                    while len(row) % 4:
                        row.append(0)
                    new_offs = list(offs)
                    new_offs[stick_tags.STR_COMMENT] = len(row)
                    row += encode(new)
                    while len(row) % 4:
                        row.append(0)
                    struct.pack_into("<21H", row, 0x5E, *new_offs)
                    free, used = struct.unpack_from("<HH", b, base + PAGE_FREE)
                    if len(row) > free:
                        skipped_full += 1
                        continue
                    start = (base + 0x28 + used + 3) // 4 * 4
                    if start + len(row) > gbase - 6 - 2 * 15 - 0x24 * (n_off // 16):    # must stay clear of the row table
                        skipped_full += 1
                        continue
                    b[start:start + len(row)] = row
                    struct.pack_into("<H", b, slot, start - (base + 0x28))
                    struct.pack_into("<HH", b, base + PAGE_FREE, free - len(row) - (start - (base + 0x28 + used)), used + len(row) + (start - (base + 0x28 + used)))
                    done += 1
        nxt = struct.unpack_from("<I", b, base + 12)[0]
        page = None if page == last else nxt
    open(dst, "wb").write(b)
    return {"edited": done, "already right": untouched, "page full, skipped": skipped_full}
