"""The corpus's GPT training step in torch, test/bench/gpt.jl's model layer
for layer: token embedding with a tied head, RMSNorm, grouped-query attention
with rotary positions (rotate-half) and a causal mask, SwiGLU, cross-entropy,
and Adam (lr 3e-3), on tokens drawn at random, TF32.

    python gpt_torch.py --vocab V --hidden D --heads H --kv-heads K --ffn F --layers L --T T --batch B

Prints one JSON line per way the step runs: eager, the loss under
torch.compile, and that compiled step, its backward and Adam's update
captured as one CUDA graph and replayed. A step's time is the best of 20
timed by CUDA events; first_step_s is the seconds to the first step's
result, torch.compile's compilation in it.
"""

import argparse
import json
import time

import torch
import torch.nn.functional as F


def rms_norm(x, weight, eps=1e-5):
    return weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def rotary(head_dim, T, theta=10000.0):
    inv_freq = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    angles = torch.outer(torch.arange(T, dtype=torch.float32), inv_freq)
    return torch.cat([angles.cos(), angles.cos()], -1).cuda(), torch.cat([-angles.sin(), angles.sin()], -1).cuda()


def rope(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat([x[..., half:], x[..., :half]], -1) * sin


def attention(x, p, cos, sin, a):
    B, T, D = x.shape
    q = (x @ p["wq"]).view(B, T, a.heads, -1).transpose(1, 2)
    k = (x @ p["wk"]).view(B, T, a.kv_heads, -1).transpose(1, 2)
    v = (x @ p["wv"]).view(B, T, a.kv_heads, -1).transpose(1, 2)
    o = F.scaled_dot_product_attention(rope(q, cos, sin), rope(k, cos, sin), v, is_causal=True,
                                       enable_gqa=a.heads != a.kv_heads)
    return o.transpose(1, 2).reshape(B, T, D) @ p["wo"]


def loss_of(params, tokens, targets, cos, sin, a):
    x = params["wte"][tokens]
    for p in params["blocks"]:
        x = x + attention(rms_norm(x, p["norm1"]), p, cos, sin, a)
        h = rms_norm(x, p["norm2"])
        x = x + (F.silu(h @ p["wg"]) * (h @ p["wu"])) @ p["wd"]
    logits = rms_norm(x, params["norm"]) @ params["wte"].T
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))


def parameters(a):
    head_dim = a.hidden // a.heads
    init = lambda *dims: torch.nn.Parameter(0.02 * torch.randn(*dims, device="cuda"))
    ones = lambda n: torch.nn.Parameter(torch.ones(n, device="cuda"))
    blocks = [dict(norm1=ones(a.hidden), wq=init(a.hidden, a.heads * head_dim), wk=init(a.hidden, a.kv_heads * head_dim),
                   wv=init(a.hidden, a.kv_heads * head_dim), wo=init(a.heads * head_dim, a.hidden), norm2=ones(a.hidden),
                   wg=init(a.hidden, a.ffn), wu=init(a.hidden, a.ffn), wd=init(a.ffn, a.hidden)) for _ in range(a.layers)]
    return dict(wte=init(a.vocab, a.hidden), blocks=blocks, norm=ones(a.hidden))


def leaves(params):
    return [params["wte"], params["norm"]] + [w for b in params["blocks"] for w in b.values()]


def best(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return min(times)


def graphed(step):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    return graph.replay


def emit(side, first_step_s, step_ms):
    print(json.dumps({"side": side, "precision": "tf32", "first_step_s": round(first_step_s, 2),
                      "step_ms": round(step_ms, 3), "note": f"torch {torch.__version__}, best of 20"}), flush=True)


def main():
    p = argparse.ArgumentParser()
    for name in ("vocab", "hidden", "heads", "kv-heads", "ffn", "layers", "T", "batch"):
        p.add_argument("--" + name, type=int, required=True)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    params = parameters(a)
    cos, sin = rotary(a.hidden // a.heads, a.T)
    tokens = torch.randint(0, a.vocab, (a.batch, a.T), device="cuda")
    targets = torch.randint(0, a.vocab, (a.batch, a.T), device="cuda")
    # Gradients stay allocated between steps, so a captured graph writes the same ones.
    opt = torch.optim.Adam(leaves(params), lr=3e-3, betas=(0.9, 0.999), eps=1e-8, foreach=True, capturable=True)

    def step(loss_fn):
        opt.zero_grad(set_to_none=False)
        loss = loss_fn(params, tokens, targets, cos, sin, a)
        loss.backward()
        opt.step()
        return loss

    for side, loss_fn in (("torch eager", loss_of), ("torch compiled", torch.compile(loss_of))):
        torch.cuda.synchronize()
        started = time.perf_counter()
        step(loss_fn)
        torch.cuda.synchronize()
        first = time.perf_counter() - started
        emit(side, first, best(lambda: step(loss_fn)))
        if side == "torch compiled":
            emit("torch compiled graph", first, best(graphed(lambda: step(loss_fn))))


if __name__ == "__main__":
    main()
