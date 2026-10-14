# Per-step kernel time by category for the steps analyze.py selects.
import sqlite3, sys, collections, re
path, steps, skip = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
names = dict(c.execute("select id, value from StringIds"))
copies = list(c.execute("select bytes, count(*) from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 group by bytes"))
marker, count = sorted((b, n) for b, n in copies if steps <= n <= 2 * steps + 4 and b >= 1024)[0]
per = count // steps
starts = [s for (s,) in c.execute("select start from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 and bytes=? order by start", (marker,))][-steps * per:][::per]
t0, t1 = starts[skip], starts[-1]; n = len(starts) - 1 - skip
rules = [
    ("matmul (XLA gemm fusion)", r"gemm_fusion|triton_gemm"),
    ("matmul (cuBLAS)", r"gemm|sgemm|xmma|cutlass|nvjet|cublas|Kernel2|splitKreduce"),
    ("attention (fused cuTile)", r"attention_"),
    ("XLA reduction fusions", r"^input_.*reduce|reduce_fusion|^input_reduce"),
    ("XLA transpose fusions", r"transpose_fusion"),
    ("XLA other fusions", r"^loop_|^input_|fusion|^wrapped|concatenate|slice|copy"),
    ("cuTile graph regions", r"graph_|cutile"),
    ("Tempest CUDA pointwise", r"pointwise|cuda_.*kernel"),
    ("Tempest reductions/softmax", r"sum_kernel|logsumexp|reduction|partials"),
    ("Tempest index/scatter", r"gather|scatter|take_rows|put_rows"),
]
tot = collections.defaultdict(lambda: [0, 0]); other = collections.Counter()
for s, e, d in c.execute("select start, end, demangledName from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<?", (t0, t1)):
    k = names.get(d, "?")
    for label, rx in rules:
        if re.search(rx, k, re.I): break
    else:
        label = "other"; other[k[:80]] += e - s
    tot[label][0] += e - s; tot[label][1] += 1
dtod = c.execute("select coalesce(sum(end-start),0), count(*) from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=8 and start>=? and start<?", (t0, t1)).fetchone()
tot["device-to-device copies"] = [dtod[0], dtod[1]]
for k, (t, m) in sorted(tot.items(), key=lambda x: -x[1][0]):
    print(f"{t/n/1e6:8.3f} ms {m/n:7.1f}/step  {k}")
print(f"{sum(v[0] for v in tot.values())/n/1e6:8.3f} ms total")
for k, t in other.most_common(8): print("   other:", round(t/n/1e3, 1), "us", k)
