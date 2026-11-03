"""Encodes a directory of PNG frames (f0000.png, ...) as a looping animated AVIF.

    python3 encode.py frames/ out.avif fps
"""
import glob
import os
import struct
import sys

from PIL import Image, features

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def boxes(data):
    """The boxes in data, as (type, payload) pairs."""
    out = []
    at = 0
    while at < len(data):
        size, kind = struct.unpack(">I4s", data[at:at + 8])
        out.append((kind, data[at + 8:at + size]))
        at += size
    return out


def box(kind, payload):
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def tables(trak):
    """A track's sample sizes and each sample's offset in the file."""
    found = {}

    def walk(data):
        for kind, payload in boxes(data):
            if kind in CONTAINERS:
                walk(payload)
            else:
                found[kind] = payload

    walk(trak)
    count, = struct.unpack(">I", found[b"stsz"][8:12])
    sizes = struct.unpack(f">{count}I", found[b"stsz"][12:12 + 4 * count])
    chunks, = struct.unpack(">I", found[b"stco"][4:8])
    starts = struct.unpack(f">{chunks}I", found[b"stco"][8:8 + 4 * chunks])
    runs, = struct.unpack(">I", found[b"stsc"][4:8])
    stsc = [struct.unpack(">III", found[b"stsc"][8 + 12 * i:20 + 12 * i]) for i in range(runs)]
    offsets = []
    sample = 0
    for chunk in range(chunks):
        # The last stsc run whose first chunk (1-based) is at or before this one says how many samples it holds.
        per_chunk = [n for first, n, _ in stsc if first <= chunk + 1][-1]
        at = starts[chunk]
        for _ in range(per_chunk):
            offsets.append(at)
            at += sizes[sample]
            sample += 1
    return list(sizes), offsets


def timeless(payload):
    """A movie, track or media header with its creation and modification times set to zero, so the same
    frames always make the same file."""
    times = 16 if payload[0] == 1 else 8
    return payload[:4] + bytes(times) + payload[4 + times:]


def retabled(data, offsets):
    """The box tree with its chunk tables replaced: one chunk per sample, at the given offsets."""
    out = b""
    for kind, payload in boxes(data):
        if kind in CONTAINERS:
            payload = retabled(payload, offsets)
        elif kind in (b"tkhd", b"mdhd"):
            payload = timeless(payload)
        elif kind == b"stco":
            payload = payload[:4] + struct.pack(f">I{len(offsets)}I", len(offsets), *offsets)
        elif kind == b"stsc":
            payload = payload[:4] + struct.pack(">IIII", 1, 1, 1, 1)
        out += box(kind, payload)
    return out


def relocated(meta, moved):
    """The meta box with its item locations moved as the samples they point at were."""
    out = b""
    for kind, payload in boxes(meta[4:]):
        if kind == b"iloc":
            payload = bytearray(payload)
            version = payload[0]
            offset_size, length_size = payload[4] >> 4, payload[4] & 15
            base_size, index_size = payload[5] >> 4, payload[5] & 15
            id_size = 4 if version == 2 else 2
            items = int.from_bytes(payload[6:6 + id_size], "big")
            at = 6 + id_size
            for _ in range(items):
                at += id_size                         # item id
                if version in (1, 2):
                    at += 2                           # construction method
                at += 2                               # data reference index
                base = int.from_bytes(payload[at:at + base_size], "big")
                at += base_size
                extents = int.from_bytes(payload[at:at + 2], "big")
                at += 2
                for _ in range(extents):
                    if version in (1, 2):
                        at += index_size
                    old = base + int.from_bytes(payload[at:at + offset_size], "big")
                    payload[at:at + offset_size] = (moved[old] - base).to_bytes(offset_size, "big")
                    at += offset_size + length_size
            payload = bytes(payload)
        out += box(kind, payload)
    return box(b"meta", meta[:4] + out)


def interleave(path):
    """Rewrites an animated AVIF so each frame's colour and alpha sit together, in frame order.

    libavif writes every alpha frame before the first colour frame, so a browser shows nothing until
    a third of the file has arrived. Interleaved, the first frame shows once its own two samples are in.
    """
    data = open(path, "rb").read()
    top = dict(boxes(data))
    traks = [payload for kind, payload in boxes(top[b"moov"]) if kind == b"trak"]
    tracks = [tables(trak) for trak in traks]
    frames = len(tracks[0][0])
    # The new chunk tables hold one entry per sample whatever the offsets, so the new moov's size is known
    # before the offsets are.
    moov_size = 8
    for kind, payload in boxes(top[b"moov"]):
        moov_size += 8 + len(retabled(payload, tracks[0][1]) if kind == b"trak" else payload)
    at = len(box(b"ftyp", top[b"ftyp"])) + len(box(b"meta", top[b"meta"])) + moov_size + 8
    new_offsets = [[0] * frames for _ in tracks]
    moved = {}
    mdat = b""
    for i in range(frames):
        for track, (sizes, offsets) in enumerate(tracks):
            new_offsets[track][i] = at
            moved[offsets[i]] = at
            mdat += data[offsets[i]:offsets[i] + sizes[i]]
            at += sizes[i]
    moov = b""
    track = 0
    for kind, payload in boxes(top[b"moov"]):
        if kind == b"trak":
            payload = retabled(payload, new_offsets[track])
            track += 1
        elif kind == b"mvhd":
            payload = timeless(payload)
        moov += box(kind, payload)
    moov = box(b"moov", moov)
    assert len(moov) == moov_size
    out = box(b"ftyp", top[b"ftyp"]) + relocated(top[b"meta"], moved) + moov + box(b"mdat", mdat)
    open(path, "wb").write(out)


src, out, fps = sys.argv[1], sys.argv[2], int(sys.argv[3])
if not features.check("avif"):
    sys.exit("encode.py: this Pillow has no AVIF support; python3 -m pip install -r requirements.txt")
frames = [Image.open(f).convert("RGBA") for f in sorted(glob.glob(os.path.join(src, "f*.png")))]
# Whole milliseconds per frame, rounded so the loop lasts exactly len(frames) / fps seconds.
durations = [round((i + 1) * 1000 / fps) - round(i * 1000 / fps) for i in range(len(frames))]
# Quality 60 keeps the faint lines while the file stays near 2 MB; speed 6 is libavif's default trade.
frames[0].save(out, "AVIF", save_all=True, append_images=frames[1:], duration=durations, loop=0, quality=60, speed=6)
interleave(out)
print(f"{out}: {os.path.getsize(out) / 1e6:.2f} MB")
