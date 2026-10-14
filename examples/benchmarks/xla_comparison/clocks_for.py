# Mean SM clock and power over the measured step window of a trace.
import sqlite3, sys, datetime
path, steps, skip = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
epoch = c.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
copies = list(c.execute("select bytes, count(*) from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 group by bytes"))
marker, count = sorted((b, n) for b, n in copies if steps <= n <= 2 * steps + 4 and b >= 1024)[0]
per = count // steps
starts = [s for (s,) in c.execute("select start from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 and bytes=? order by start", (marker,))][-steps * per:][::per]
t0, t1 = (epoch + starts[skip]) / 1e9, (epoch + starts[-1]) / 1e9
mhz, watts = [], []
for line in open("clocks.log"):
    ts, clk, pw, util = [x.strip() for x in line.split(",")]
    t = datetime.datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").timestamp()
    if t0 <= t <= t1:
        mhz.append(float(clk.split()[0])); watts.append(float(pw.split()[0]))
print(f"{path}: {len(mhz)} samples, SM {sum(mhz)/max(len(mhz),1):.0f} MHz (min {min(mhz, default=0):.0f}, max {max(mhz, default=0):.0f}), {sum(watts)/max(len(watts),1):.1f} W")
