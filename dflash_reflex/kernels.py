"""Triton block Reflex kernels, adapted from SpecNaacl (see SOURCE_MANIFEST).

No autograd, device scalar extraction, or vocabulary-sized probability cache.
The input logits and teacher probabilities already exist in vLLM.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scan(
    Z,
    U,
    B,
    ACTIVE,
    MAX,
    SUM,
    VALUES,
    IDS,
    V: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    TILES: tl.constexpr,
    BV: tl.constexpr,
    MODE: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    token = tile * BV + tl.arange(0, BV)
    raw = tl.load(Z + row * V + token, token < V, other=-float("inf")).to(tl.float32)
    if MODE != 0:
        live = token < V
        if MODE == 1:
            live = live & (tl.load(ACTIVE + token, token < V, other=0) != 0)
        correction = tl.full((BV,), 0.0, tl.float32)
        for r in tl.static_range(R):
            correction += tl.load(U + row * R + r) * tl.load(B + token * R + r, live, other=0)
        raw += correction
    maximum = tl.max(raw, 0)
    shift = tl.where(maximum == -float("inf"), 0.0, maximum)
    mass = tl.where(token < V, tl.exp(raw - shift), 0.0)
    # Preserve the source's fixed FP32 pairwise normalization tree.
    for level in tl.static_range(8):
        mass = tl.sum(tl.reshape(mass, (BV // (1 << (level + 1)), 2)), 1)
    index = row * TILES + tile
    tl.store(MAX + index, maximum)
    tl.store(SUM + index, tl.sum(mass, 0))
    live = token < V
    for j in range(K):
        best = tl.max(raw, 0)
        chosen = tl.min(tl.where((raw == best) & live, token, V), 0)
        tl.store(VALUES + index * K + j, best)
        tl.store(IDS + index * K + j, chosen)
        raw = tl.where(token == chosen, -float("inf"), raw)
        live = live & (token != chosen)


@triton.jit
def _merge(
    MAX,
    SUM,
    VALUES,
    IDS,
    Q,
    OUT_IDS,
    NORM,
    V: tl.constexpr,
    K: tl.constexpr,
    TILES: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    t = tl.arange(0, BT)
    maxima = tl.load(MAX + row * TILES + t, t < TILES, other=-float("inf"))
    sums = tl.load(SUM + row * TILES + t, t < TILES, other=0)
    maximum = tl.max(maxima, 0)
    total = tl.sum(sums * tl.exp(maxima - maximum), 0)
    tl.store(NORM + row * 2, maximum)
    tl.store(NORM + row * 2 + 1, total)
    c = tl.arange(0, BK)
    z = tl.load(VALUES + row * TILES * K + c, c < TILES * K, other=-float("inf"))
    ids = tl.load(IDS + row * TILES * K + c, c < TILES * K, other=V)
    for j in range(K):
        best = tl.max(z, 0)
        chosen = tl.min(tl.where(z == best, ids, V), 0)
        tl.store(OUT_IDS + row * K + j, chosen)
        tl.store(Q + row * K + j, tl.div_rn(tl.exp(best - maximum), total))
        z = tl.where(ids == chosen, -float("inf"), z)


class ProposalWorkspace:
    def __init__(self, rows, vocab, rank, topk, device):
        self.rows, self.vocab, self.topk = rows, vocab, min(topk, vocab)
        tiles = triton.cdiv(vocab, 256)
        self.maxima = torch.empty(rows, tiles, device=device)
        self.sums = torch.empty_like(self.maxima)
        self.values = torch.empty(rows, tiles, self.topk, device=device)
        self.indices = torch.empty(rows, tiles, self.topk, device=device, dtype=torch.long)
        self.q = torch.empty(rows, self.topk, device=device)
        self.ids = torch.empty(rows, self.topk, device=device, dtype=torch.long)
        self.norm = torch.empty(rows, 2, device=device)
        self.u = torch.empty(rows, rank, device=device)
        self.dense = None  # Allocated only if an explicit GEMM is selected.


@torch.no_grad()
def propose(adapter, hidden, raw, workspace):
    n, vocab = raw.shape
    if n > workspace.rows or not raw.is_contiguous():
        raise ValueError("proposal workspace or contiguous logits contract violated")
    k, rank = workspace.topk, adapter.config.rank
    with adapter.phase("logit_correction_ms"):
        u = workspace.u[:n]
        # FP32 input is small (slots*hidden), never slots*vocab*rank.
        torch.mm(hidden.float(), adapter.A, out=u)
        selected = adapter.config.proposal_mode
        dense_impl = adapter.config.dense_implementation
        key = f"{adapter.device_name}|{n}|{hidden.shape[-1]}|{vocab}|{rank}|{k}|{hidden.dtype}"
        calibration = adapter.dispatch_cache.get(key)
        choice = (
            min(calibration, key=lambda c: abs(c["density"] - adapter.host_active_bound / vocab))
            if calibration
            else None
        )
        if selected == "auto":
            selected = (
                choice["mode"]
                if choice
                else ("sparse" if adapter.host_active_bound < vocab // 8 else "dense")
            )
        if dense_impl == "auto" and choice and choice["mode"] == "dense":
            dense_impl = choice["implementation"]
        mode = 0 if not adapter.has_updated else (1 if selected == "sparse" else 2)
        if mode == 1:
            adapter.sparse_dispatch_count += 1
        elif mode == 2:
            adapter.dense_dispatch_count += 1
        if mode == 2 and dense_impl == "gemm":
            if workspace.dense is None:
                workspace.dense = torch.empty(workspace.rows, vocab, device=raw.device)
            corrected = workspace.dense[:n]
            torch.mm(u, adapter.B.t(), out=corrected)
            corrected.add_(raw)
            raw, mode = corrected, 0
    with adapter.phase("proposal_sampling_ms"):
        tiles = triton.cdiv(vocab, 256)
        _scan[(n, tiles)](
            raw,
            u,
            adapter.B,
            adapter.active,
            workspace.maxima,
            workspace.sums,
            workspace.values,
            workspace.indices,
            vocab,
            rank,
            k,
            tiles,
            256,
            mode,
            enable_fp_fusion=False,
            num_warps=4,
        )
        _merge[(n,)](
            workspace.maxima,
            workspace.sums,
            workspace.values,
            workspace.indices,
            workspace.q,
            workspace.ids,
            workspace.norm,
            vocab,
            k,
            tiles,
            triton.next_power_of_2(tiles),
            triton.next_power_of_2(tiles * k),
            enable_fp_fusion=False,
            num_warps=4,
        )
    return workspace.ids[:n], workspace.q[:n], workspace.norm[:n], u


@triton.jit
def _target_q(
    TIDS,
    DIDS,
    WEIGHTS,
    H,
    HEAD,
    B,
    U,
    NORM,
    Q,
    D: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    BH: tl.constexpr,
    BR: tl.constexpr,
):
    row, slot = tl.program_id(0), tl.program_id(1)
    token = tl.load(TIDS + row * K + slot)
    other = tl.arange(0, triton.next_power_of_2(K))
    draft = tl.load(DIDS + row * K + other, other < K, other=-1)
    if tl.load(WEIGHTS + row) <= 0 or tl.sum((draft == token).to(tl.int32), 0) > 0:
        tl.store(Q + row * K + slot, 0.0)
        return
    h, r = tl.arange(0, BH), tl.arange(0, BR)
    x = tl.load(H + row * D + h, h < D, other=0).to(tl.float32)
    head = tl.load(HEAD + token * D + h, h < D, other=0).to(tl.float32)
    raw = tl.sum(head * x, 0).to(HEAD.dtype.element_ty).to(tl.float32)
    b = tl.load(B + token * R + r, r < R, other=0)
    u = tl.load(U + row * R + r, r < R, other=0)
    z = raw + tl.sum(b * u, 0)
    maximum, total = tl.load(NORM + row * 2), tl.load(NORM + row * 2 + 1)
    tl.store(Q + row * K + slot, tl.div_rn(tl.exp(z - maximum), total))


@triton.jit
def _union_terms(
    P,
    TIDS,
    TQ,
    B,
    U,
    DIDS,
    DQ,
    WEIGHTS,
    OUT_IDS,
    OUT_G,
    VEC,
    STATS,
    V: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    BU: tl.constexpr,
    BR: tl.constexpr,
):
    row = tl.program_id(0)
    j, r = tl.arange(0, BU), tl.arange(0, BR)
    if tl.load(WEIGHTS + row) <= 0:
        tl.store(OUT_IDS + row * 2 * K + j, -1, j < 2 * K)
        tl.store(OUT_G + row * 2 * K + j, 0.0, j < 2 * K)
        tl.store(VEC + row * R + r, 0.0, r < R)
        s = tl.arange(0, 4)
        tl.store(STATS + row * 4 + s, 0.0)
        return
    dk = tl.arange(0, triton.next_power_of_2(K))
    di = tl.load(DIDS + row * K + dk, dk < K, other=-1)
    first = j < K
    ids = tl.where(
        first,
        tl.load(DIDS + row * K + j % K, j < 2 * K, other=0),
        tl.load(TIDS + row * K + j % K, j < 2 * K, other=0),
    )
    p = tl.load(P + row * V + ids, (j < 2 * K) & (ids < V), other=0)
    overlap = tl.sum((ids[:, None] == di[None, :]).to(tl.int32), 1) > 0
    valid = (j < 2 * K) & (first | ((~overlap) & (p > 0))) & (ids < V)
    b = tl.load(B + ids[:, None] * R + r[None, :], valid[:, None] & (r[None, :] < R), other=0)
    q = tl.where(
        first,
        tl.load(DQ + row * K + j % K, j < 2 * K, other=0),
        tl.load(TQ + row * K + j % K, j < 2 * K, other=0),
    )
    q, p = tl.where(valid, q, 0), tl.where(valid, p, 0)
    full = tl.sum(valid.to(tl.int32), 0) == V
    pt = tl.where(full, 0.0, tl.maximum(1.0 - tl.sum(p, 0), 0.0))
    qt = tl.where(full, 0.0, tl.maximum(1.0 - tl.sum(q, 0), 0.0))
    kl = tl.sum(tl.where(p > 0, p * (tl.log(p) - tl.log(q)), 0.0), 0)
    kl += tl.where(pt > 0, pt * (tl.log(pt) - tl.log(qt)), 0.0)
    tv = 0.5 * (tl.sum(tl.abs(p - q), 0) + tl.abs(pt - qt))
    weight = tl.load(WEIGHTS + row)
    g = (q - p) * weight
    tl.store(OUT_IDS + row * 2 * K + j, tl.where(valid, ids, -1), j < 2 * K)
    tl.store(OUT_G + row * 2 * K + j, g, j < 2 * K)
    # A terms are explicitly evaluated against B_t, before _update launches.
    v = tl.sum(g[:, None] * b, 0)
    tl.store(VEC + row * R + r, v, r < R)
    tl.store(STATS + row * 4, weight)
    tl.store(STATS + row * 4 + 1, tl.where(weight > 0, kl * weight, 0))
    tl.store(STATS + row * 4 + 2, tl.where(weight > 0, tv * weight, 0))
    tl.store(STATS + row * 4 + 3, (weight > 0).to(tl.float32))


@triton.jit
def _update(
    IDS,
    G,
    U,
    DEN,
    B,
    ACTIVE,
    ACTIVE_IDS,
    ACTIVE_COUNT,
    R: tl.constexpr,
    K: tl.constexpr,
    LR: tl.constexpr,
    BU: tl.constexpr,
    BR: tl.constexpr,
):
    row = tl.program_id(0)
    j, r = tl.arange(0, BU), tl.arange(0, BR)
    ids = tl.load(IDS + row * 2 * K + j, j < 2 * K, other=-1)
    g = tl.load(G + row * 2 * K + j, j < 2 * K, other=0)
    u = tl.load(U + row * R + r, r < R, other=0)
    den = tl.load(DEN)
    delta = -LR * tl.div_rn(g[:, None] * u[None, :], tl.where(den > 0, den, 1.0))
    valid = (ids >= 0) & (j < 2 * K) & (den > 0)
    tl.atomic_add(
        B + tl.maximum(ids[:, None], 0) * R + r[None, :], delta, valid[:, None] & (r[None, :] < R)
    )
    touched = valid & (tl.sum((delta != 0).to(tl.int32), 1) > 0)
    old = tl.atomic_xchg(ACTIVE + tl.maximum(ids, 0), 1, touched)
    first = touched & (old == 0)
    slot = tl.atomic_add(ACTIVE_COUNT + tl.zeros((BU,), tl.int32), 1, first)
    tl.store(ACTIVE_IDS + slot, ids, first)


class FeedbackWorkspace:
    def __init__(self, rows, hidden, rank, topk, device):
        self.ids = torch.empty(rows, 2 * topk, device=device, dtype=torch.long)
        self.g = torch.empty(rows, 2 * topk, device=device)
        self.v = torch.empty(rows, rank, device=device)
        self.stats = torch.empty(rows, 4, device=device)
        self.delta_a = torch.empty(hidden, rank, device=device)
        self.target_q = torch.empty(rows, topk, device=device)


@torch.no_grad()
def feedback(adapter, hidden, u, draft_ids, draft_q, norm, p, weights, workspace):
    adapter.round += 1
    if adapter.round % adapter.config.update_interval:
        return
    n, vocab = p.shape
    k, rank, hidden_size = draft_ids.shape[-1], adapter.config.rank, hidden.shape[-1]
    with adapter.phase("topk_union_ms"):
        # The sampler already computed normalized, constrained p; no re-softmax.
        teacher_ids = p.topk(k, dim=-1, sorted=True).indices
        _target_q[(n, k)](
            teacher_ids,
            draft_ids,
            weights,
            hidden,
            adapter.head,
            adapter.B,
            u,
            norm,
            workspace.target_q,
            hidden_size,
            rank,
            k,
            triton.next_power_of_2(hidden_size),
            triton.next_power_of_2(rank),
            num_warps=4,
            enable_fp_fusion=False,
        )
        _union_terms[(n,)](
            p,
            teacher_ids,
            workspace.target_q,
            adapter.B,
            u,
            draft_ids,
            draft_q,
            weights,
            workspace.ids,
            workspace.g,
            workspace.v,
            workspace.stats,
            vocab,
            rank,
            k,
            triton.next_power_of_2(2 * k),
            triton.next_power_of_2(rank),
            num_warps=4,
            enable_fp_fusion=False,
        )
        sums = workspace.stats[:n].sum(0)
    with adapter.phase("A_gradient_ms"):
        torch.mm(hidden.float().t(), workspace.v[:n], out=workspace.delta_a)
        adapter.grad_sum.add_(workspace.delta_a)
        adapter.grad_weight.add_(sums[0])
    with adapter.phase("B_update_ms"):
        _update[(n,)](
            workspace.ids,
            workspace.g,
            u,
            sums[:1],
            adapter.B,
            adapter.active,
            adapter.active_ids,
            adapter.active_count,
            rank,
            k,
            adapter.config.fast_lr,
            triton.next_power_of_2(2 * k),
            triton.next_power_of_2(rank),
            enable_fp_fusion=False,
            num_warps=4,
        )
    adapter.has_updated = True
    adapter.host_active_bound = min(vocab, adapter.host_active_bound + n * 2 * k)
    return sums
