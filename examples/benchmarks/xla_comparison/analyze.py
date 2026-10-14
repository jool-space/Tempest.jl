# Split an nsys trace into training steps by the per-step host-to-device batch copy,
# then report per-step wall time, GPU busy time and kernel time by name.
import sqlite3, sys, re, collections
path, steps, skip = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
import os
assert os.path.getsize(path) > 0, "trace not exported yet"
c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
names = dict(c.execute("select id, value from StringIds"))
copies = list(c.execute("select bytes, count(*) from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 group by bytes"))
# the marker: the smallest host-to-device copy size that occurs once per step (or once per step plus preparation)
candidates = sorted((b, n) for b, n in copies if steps <= n <= 2 * steps + 4 and b >= 1024)
marker, count = candidates[0]
per = count // steps
starts = [s for (s,) in c.execute("select start from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 and bytes=? order by start", (marker,))]
starts = starts[-steps * per:][::per]
bounds = list(zip(starts[skip:-1], starts[skip + 1:]))
t0, t1 = bounds[0][0], bounds[-1][1]
n = len(bounds)
kernels = list(c.execute("select start, end, demangledName, shortName from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<? order by start", (t0, t1)))
wall = (t1 - t0) / n
busy = 0; last = t0
for s, e, _, _ in kernels:
    s = max(s, last)
    if e > s: busy += e - s; last = e
by = collections.defaultdict(lambda: [0, 0])
for s, e, d, sh in kernels:
    k = names.get(d, "?")
    by[k][0] += e - s; by[k][1] += 1
print(f"marker {marker} B x{per}; {n} steps; wall {wall/1e6:.3f} ms/step; GPU busy {busy/n/1e6:.3f} ms; idle {(wall - busy/n)/1e6:.3f} ms; kernels/step {len(kernels)/n:.0f}")
total = sum(v[0] for v in by.values())
print(f"kernel time {total/n/1e6:.3f} ms/step")
for k, (t, m) in sorted(by.items(), key=lambda x: -x[1][0])[:int(sys.argv[4]) if len(sys.argv) > 4 else 40]:
    print(f"{t/n/1e3:10.1f} us {m/n:6.1f}x  {k[:150]}")
