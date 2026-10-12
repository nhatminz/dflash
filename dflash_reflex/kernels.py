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
    WEIGHTS=None,
    ROWS=None,
    ACTIVE_COUNT=None,
    THRESHOLD: tl.constexpr = 0,
    NORMALIZE: tl.constexpr = True,
    DISPATCH_COUNTS=None,
    COUNT_DENSE: tl.constexpr = False,
    VALID_COUNT=None,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    if VALID_COUNT is not None:
        if row >= tl.load(VALID_COUNT):
            return
    if DISPATCH_COUNTS is not None:
        if (row == 0) & (tile == 0) & (MODE != 0 or COUNT_DENSE):
            dense = tl.full((), (MODE == 2) or COUNT_DENSE, tl.int32)
            if MODE == 3:
                dense = tl.load(ACTIVE_COUNT) >= THRESHOLD
            tl.atomic_add(DISPATCH_COUNTS + dense.to(tl.int32), 1)
    token = tile * BV + tl.arange(0, BV)
    if WEIGHTS is not None:
        if tl.load(WEIGHTS + row) <= 0:
            return
    source = row
    if ROWS is not None:
        source = tl.load(ROWS + row)
    raw = tl.load(Z + source * V + token, token < V, other=-float("inf")).to(tl.float32)
    if MODE != 0:
        live = token < V
        if MODE == 1:
            live = live & (tl.load(ACTIVE + token, token < V, other=0) != 0)
        if MODE == 3:
            if tl.load(ACTIVE_COUNT) < THRESHOLD:
                live = live & (tl.load(ACTIVE + token, token < V, other=0) != 0)
        correction = tl.full((BV,), 0.0, tl.float32)
        for r in tl.static_range(R):
            correction += tl.load(U + row * R + r) * tl.load(B + token * R + r, live, other=0)
        raw += correction
    maximum = tl.max(raw, 0)
    shift = tl.where(maximum == -float("inf"), 0.0, maximum)
    mass = tl.full((BV,), 0.0, tl.float32)
    if NORMALIZE:
        mass = tl.where(token < V, tl.exp(raw - shift), 0.0)
    # Preserve the source's fixed FP32 pairwise normalization tree.
    for level in tl.static_range(8):
        mass = tl.sum(tl.reshape(mass, (BV // (1 << (level + 1)), 2)), 1)
    index = row * TILES + tile
    if NORMALIZE:
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
    WEIGHTS=None,
    FALLBACK_IDS=None,
    NORMALIZE: tl.constexpr = True,
    OUT_ROWS=None,
    VALID_COUNT=None,
):
    row = tl.program_id(0)
    if VALID_COUNT is not None:
        if row >= tl.load(VALID_COUNT):
            return
    output_row = row
    if OUT_ROWS is not None:
        output_row = tl.load(OUT_ROWS + row)
    if WEIGHTS is not None:
        if tl.load(WEIGHTS + row) <= 0:
            j = tl.arange(0, triton.next_power_of_2(K))
            fallback = tl.full((triton.next_power_of_2(K),), 0, tl.int64)
            if FALLBACK_IDS is not None:
                fallback = tl.load(FALLBACK_IDS + row * K + j, j < K, other=0)
            tl.store(OUT_IDS + row * K + j, fallback, j < K)
            tl.store(Q + row * K + j, 0.0, j < K)
            tl.store(NORM + row * 2, 0.0)
            tl.store(NORM + row * 2 + 1, 1.0)
            return
    maximum, total = 0.0, 1.0
    if NORMALIZE:
        t = tl.arange(0, BT)
        maxima = tl.load(MAX + row * TILES + t, t < TILES, other=-float("inf"))
        sums = tl.load(SUM + row * TILES + t, t < TILES, other=0)
        maximum = tl.max(maxima, 0)
        total = tl.sum(sums * tl.exp(maxima - maximum), 0)
    if NORMALIZE:
        tl.store(NORM + row * 2, maximum)
        tl.store(NORM + row * 2 + 1, total)
    c = tl.arange(0, BK)
    z = tl.load(VALUES + row * TILES * K + c, c < TILES * K, other=-float("inf"))
    ids = tl.load(IDS + row * TILES * K + c, c < TILES * K, other=V)
    for j in range(K):
        best = tl.max(z, 0)
        chosen = tl.min(tl.where(z == best, ids, V), 0)
        tl.store(OUT_IDS + output_row * K + j, chosen)
        if NORMALIZE:
            tl.store(Q + row * K + j, tl.div_rn(tl.exp(best - maximum), total))
        z = tl.where(ids == chosen, -float("inf"), z)


class ProposalWorkspace:
    @torch.inference_mode(False)
    def __init__(self, rows, vocab, rank, topk, device, tile_size=256):
        self.rows, self.vocab, self.topk = rows, vocab, min(topk, vocab)
        tiles = triton.cdiv(vocab, tile_size)
        self.maxima = torch.empty(rows, tiles, device=device)
        self.argmax_ids = torch.empty(rows, tiles, device=device, dtype=torch.long)
        self.sums = torch.empty_like(self.maxima)
        self.values = torch.empty(rows, tiles, self.topk, device=device)
        self.indices = torch.empty(rows, tiles, self.topk, device=device, dtype=torch.long)
        self.q = torch.empty(rows, self.topk, device=device)
        self.ids = torch.empty(rows, self.topk, device=device, dtype=torch.long)
        self.norm = torch.empty(rows, 2, device=device)
        self.u = torch.empty(rows, rank, device=device)
        self.dense = None  # Allocated only if an explicit GEMM is selected.
        self.corrected_raw = None
        self.hidden_fp32 = None


def project_hidden(adapter, hidden, workspace):
    source = hidden
    if hidden.dtype != torch.float32:
        if workspace.hidden_fp32 is None or workspace.hidden_fp32.shape[1] != hidden.shape[1]:
            workspace.hidden_fp32 = torch.empty(
                workspace.rows, hidden.shape[1], device=hidden.device, dtype=torch.float32
            )
        source = workspace.hidden_fp32[: hidden.shape[0]]
        source.copy_(hidden)
    u = workspace.u[: hidden.shape[0]]
    torch.mm(source, adapter.A, out=u)
    return u


@torch.no_grad()
def propose(adapter, hidden, raw, workspace):
    n, vocab = raw.shape
    if n > workspace.rows or not raw.is_contiguous():
        raise ValueError("proposal workspace or contiguous logits contract violated")
    k, rank = workspace.topk, adapter.config.rank
    workspace.corrected_raw = None
    with adapter.phase("logit_correction_ms"):
        u = project_hidden(adapter, hidden, workspace)
        selected = adapter.config.proposal_mode
        dense_impl = adapter.config.dense_implementation
        # Legacy caches used a host upper bound and cannot select safely.
        # An uncalibrated GPU stays sparse; no invented density crossover.
        threshold = dispatch_threshold(adapter, n, hidden.shape[-1], vocab, hidden.dtype, raw.dtype)
        if selected == "auto":
            mode = 3 if adapter.has_updated else 0
        else:
            mode = 0 if not adapter.has_updated else (1 if selected == "sparse" else 2)
        if mode == 1:
            adapter.sparse_dispatch_count += 1
        elif mode == 2:
            adapter.dense_dispatch_count += 1
        gemm_selected = mode == 2 and dense_impl == "gemm"
        if gemm_selected:
            if workspace.dense is None:
                workspace.dense = torch.empty(workspace.rows, vocab, device=raw.device)
            corrected = workspace.dense[:n]
            torch.mm(u, adapter.B.t(), out=corrected)
            corrected.add_(raw)
            workspace.corrected_raw = corrected
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
            ACTIVE_COUNT=adapter.active_count,
            THRESHOLD=threshold,
            DISPATCH_COUNTS=adapter.dispatch_counts,
            COUNT_DENSE=gemm_selected,
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
    RAW=None,
    RAW_ROWS=None,
    V: tl.constexpr = 0,
    P=None,
):
    row, slot = tl.program_id(0), tl.program_id(1)
    if tl.load(WEIGHTS + row) <= 0:
        tl.store(Q + row * K + slot, 0.0)
        return
    token = tl.load(TIDS + row * K + slot)
    if token < 0:
        tl.store(Q + row * K + slot, 0.0)
        return
    if P is not None:
        if tl.load(P + row * V + token) <= 0:
            tl.store(Q + row * K + slot, 0.0)
            return
    other = tl.arange(0, triton.next_power_of_2(K))
    draft = tl.load(DIDS + row * K + other, other < K, other=-1)
    if tl.load(WEIGHTS + row) <= 0 or tl.sum((draft == token).to(tl.int32), 0) > 0:
        tl.store(Q + row * K + slot, 0.0)
        return
    if RAW is not None:
        source = row
        if RAW_ROWS is not None:
            source = tl.load(RAW_ROWS + row)
        raw = tl.load(RAW + source * V + token).to(tl.float32)
    else:
        h = tl.arange(0, BH)
        x = tl.load(H + row * D + h, h < D, other=0).to(tl.float32)
        head = tl.load(HEAD + token * D + h, h < D, other=0).to(tl.float32)
        raw = tl.sum(head * x, 0).to(HEAD.dtype.element_ty).to(tl.float32)
    correction = tl.full((), 0.0, tl.float32)
    # Same addition order as proposal, including no contraction.
    for r in tl.static_range(R):
        correction += tl.load(B + token * R + r) * tl.load(U + row * R + r)
    z = raw + correction
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
    GREEDY: tl.constexpr = False,
    RAW=None,
    RAW_ROWS=None,
    NORM=None,
    CORRECTED_RAW: tl.constexpr = False,
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
    if GREEDY:
        chosen = tl.load(TIDS + row * K)
        p = (ids == chosen).to(tl.float32)
    else:
        p = tl.load(P + row * V + ids, (j < 2 * K) & (ids >= 0) & (ids < V), other=0)
    overlap = tl.sum((ids[:, None] == di[None, :]).to(tl.int32), 1) > 0
    valid = (j < 2 * K) & (first | ((~overlap) & (p > 0))) & (ids >= 0) & (ids < V)
    b = tl.load(B + ids[:, None] * R + r[None, :], valid[:, None] & (r[None, :] < R), other=0)
    if RAW is not None:
        source = row
        if RAW_ROWS is not None:
            source = tl.load(RAW_ROWS + row)
        raw = tl.load(RAW + source * V + ids, valid & (~first), other=0).to(tl.float32)
        z = raw
        if not CORRECTED_RAW:
            correction = tl.full((BU,), 0.0, tl.float32)
            for rr in tl.static_range(R):
                # Reuse B_t already loaded for the A-gradient term. Selecting
                # one rank preserves the proposal's serial FP32 sum order.
                coordinate = tl.sum(tl.where(r[None, :] == rr, b, 0.0), 1)
                correction += coordinate * tl.load(U + row * R + rr)
            z += correction
        maximum, total = tl.load(NORM + row * 2), tl.load(NORM + row * 2 + 1)
        teacher_q = tl.div_rn(tl.exp(z - maximum), total)
    else:
        teacher_q = tl.load(TQ + row * K + j % K, j < 2 * K, other=0)
    q = tl.where(first, tl.load(DQ + row * K + j % K, j < 2 * K, other=0), teacher_q)
    q, p = tl.where(valid, q, 0), tl.where(valid, p, 0)
    full = tl.sum(valid.to(tl.int32), 0) == V
    pt = tl.where(full, 0.0, tl.maximum(1.0 - tl.sum(p, 0), 0.0))
    qt = tl.where(full, 0.0, tl.maximum(1.0 - tl.sum(q, 0), 0.0))
    kl = tl.sum(
        tl.where(p > 0, p * (tl.log(p) - tl.log(tl.maximum(q, 1.1754943508222875e-38))), 0.0), 0
    )
    kl += tl.where(pt > 0, pt * (tl.log(pt) - tl.log(tl.maximum(qt, 1.1754943508222875e-38))), 0.0)
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
    UPDATE_COUNT=None,
):
    row = tl.program_id(0)
    j, r = tl.arange(0, BU), tl.arange(0, BR)
    ids = tl.load(IDS + row * 2 * K + j, j < 2 * K, other=-1)
    g = tl.load(G + row * 2 * K + j, j < 2 * K, other=0)
    u = tl.load(U + row * R + r, r < R, other=0)
    den = tl.load(DEN)
    if UPDATE_COUNT is not None:
        if row == 0:
            tl.atomic_add(UPDATE_COUNT, (den > 0).to(tl.int32))
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
    @torch.inference_mode(False)
    def __init__(self, rows, hidden, rank, topk, device):
        self.ids = torch.empty(rows, 2 * topk, device=device, dtype=torch.long)
        self.g = torch.empty(rows, 2 * topk, device=device)
        self.v = torch.empty(rows, rank, device=device)
        self.stats = torch.empty(rows, 4, device=device)
        self.delta_a = torch.empty(hidden, rank, device=device)
        self.target_q = torch.empty(rows, topk, device=device)
        self.teacher = None
        self.teacher_flags = torch.empty(rows, device=device)
        self.teacher_sorted = torch.empty(rows, topk, device=device, dtype=torch.long)
        self.rows, self.rank, self.topk, self.device = rows, rank, topk, device
        self.compact_rows = torch.empty(rows, device=device, dtype=torch.int32)
        self.valid_count = torch.empty((), device=device, dtype=torch.int32)
        self.greedy_teacher = None


@triton.jit
def _compact_teacher_rows(WEIGHTS, ROWS, COUNT, N: tl.constexpr, BN: tl.constexpr):
    row = tl.arange(0, BN)
    live = tl.load(WEIGHTS + row, row < N, other=0) > 0
    offsets = tl.cumsum(live.to(tl.int32)) - 1
    tl.store(ROWS + offsets, row, live)
    tl.store(COUNT, tl.sum(live.to(tl.int32), 0))


@torch.no_grad()
def feedback(
    adapter,
    hidden,
    u,
    draft_ids,
    draft_q,
    norm,
    p,
    weights,
    workspace,
    *,
    raw=None,
    raw_rows=None,
    proposal_workspace=None,
    selective=False,
    masked_teacher=False,
    greedy_teacher=False,
):
    adapter.round += 1
    if adapter.round % adapter.config.update_interval:
        return
    n, vocab = p.shape
    k, rank, hidden_size = draft_ids.shape[-1], adapter.config.rank, hidden.shape[-1]
    corrected_raw = proposal_workspace is not None and proposal_workspace.corrected_raw is not None
    if corrected_raw:
        if selective:
            raise ValueError("selective feedback cannot use GEMM proposal statistics")
        # Explicit GEMM changes the reduction order. Reuse the exact result
        # consumed by proposal instead of reconstructing its correction.
        raw = proposal_workspace.corrected_raw
    with adapter.phase("topk_union_ms"):
        # The sampler already computed normalized, constrained p; no re-softmax.
        if selective:
            if proposal_workspace is None or raw is None:
                raise ValueError("selective feedback requires exact captured logits/workspace")
            with adapter.phase("feedback_statistics_ms"):
                statistics(adapter, u, raw, proposal_workspace, weights, raw_rows)
            draft_ids = proposal_workspace.ids[:n]
            draft_q = proposal_workspace.q[:n]
            norm = proposal_workspace.norm[:n]
        with adapter.phase("feedback_topk_ms"):
            if greedy_teacher:
                teacher_ids = workspace.teacher_sorted[:n]
                teacher_ids.fill_(-1)
                tiles = triton.cdiv(vocab, 1024)
                if workspace.greedy_teacher is None:
                    workspace.greedy_teacher = (
                        torch.empty(workspace.rows, tiles, device=p.device),
                        torch.empty(workspace.rows, tiles, device=p.device, dtype=torch.long),
                    )
                values, indices = workspace.greedy_teacher
                _argmax_scan[(n, tiles)](
                    p,
                    u,
                    adapter.B,
                    adapter.active,
                    adapter.active_count,
                    values,
                    indices,
                    vocab,
                    rank,
                    tiles,
                    0,
                    0,
                    1024,
                    WEIGHTS=weights,
                )
                _argmax_merge[(n,)](
                    values,
                    indices,
                    teacher_ids,
                    vocab,
                    k,
                    tiles,
                    triton.next_power_of_2(tiles),
                    WEIGHTS=weights,
                )
            else:
                if workspace.teacher is None:
                    workspace.teacher = ProposalWorkspace(
                        workspace.rows, vocab, rank, k, p.device, tile_size=2048
                    )
                    workspace.teacher.ids = workspace.teacher_sorted
                teacher = workspace.teacher
                # Both variants skip zero weights BEFORE any vocabulary read.
                # Compaction stays on the GPU; launches have a fixed capacity.
                compact = not masked_teacher
                if compact:
                    _compact_teacher_rows[(1,)](
                        weights,
                        workspace.compact_rows,
                        workspace.valid_count,
                        n,
                        triton.next_power_of_2(n),
                    )
                tiles = triton.cdiv(vocab, 2048)
                _scan[(n, tiles)](
                    p,
                    u,
                    adapter.B,
                    adapter.active,
                    teacher.maxima,
                    teacher.sums,
                    teacher.values,
                    teacher.indices,
                    vocab,
                    rank,
                    k,
                    tiles,
                    2048,
                    0,
                    WEIGHTS=None if compact else weights,
                    ROWS=workspace.compact_rows if compact else None,
                    VALID_COUNT=workspace.valid_count if compact else None,
                    NORMALIZE=False,
                    enable_fp_fusion=False,
                )
                _merge[(n,)](
                    teacher.maxima,
                    teacher.sums,
                    teacher.values,
                    teacher.indices,
                    teacher.q,
                    teacher.ids,
                    teacher.norm,
                    vocab,
                    k,
                    tiles,
                    triton.next_power_of_2(tiles),
                    triton.next_power_of_2(tiles * k),
                    WEIGHTS=None if compact else weights,
                    OUT_ROWS=workspace.compact_rows if compact else None,
                    VALID_COUNT=workspace.valid_count if compact else None,
                    NORMALIZE=False,
                    enable_fp_fusion=False,
                )
                teacher_ids = teacher.ids[:n]
        with adapter.phase("union_kl_ms"):
            if raw is None:
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
                    RAW=raw,
                    RAW_ROWS=raw_rows,
                    V=vocab,
                    P=None if greedy_teacher else p,
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
                GREEDY=greedy_teacher,
                RAW=raw,
                RAW_ROWS=raw_rows,
                NORM=norm,
                CORRECTED_RAW=corrected_raw,
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
            UPDATE_COUNT=adapter.update_count,
            enable_fp_fusion=False,
            num_warps=4,
        )
    adapter.has_updated = True
    adapter.revision += 1
    return sums


@triton.jit
def _argmax_scan(
    Z,
    U,
    B,
    ACTIVE,
    COUNT,
    VALUES,
    IDS,
    V: tl.constexpr,
    R: tl.constexpr,
    TILES: tl.constexpr,
    MODE: tl.constexpr,
    THRESHOLD: tl.constexpr,
    BV: tl.constexpr,
    DISPATCH_COUNTS=None,
    WEIGHTS=None,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    if WEIGHTS is not None:
        if tl.load(WEIGHTS + row) <= 0:
            return
    if DISPATCH_COUNTS is not None:
        if (row == 0) & (tile == 0) & (MODE != 0):
            dense = tl.full((), MODE == 2, tl.int32)
            if MODE == 3:
                dense = tl.load(COUNT) >= THRESHOLD
            tl.atomic_add(DISPATCH_COUNTS + dense.to(tl.int32), 1)
    token = tile * BV + tl.arange(0, BV)
    z = tl.load(Z + row * V + token, token < V, other=-float("inf")).to(tl.float32)
    if MODE != 0:
        live = token < V
        if MODE == 1:
            live &= tl.load(ACTIVE + token, token < V, other=0) != 0
        if MODE == 3:
            if tl.load(COUNT) < THRESHOLD:
                live &= tl.load(ACTIVE + token, token < V, other=0) != 0
        correction = tl.full((BV,), 0.0, tl.float32)
        for r in tl.static_range(R):
            correction += tl.load(U + row * R + r) * tl.load(B + token * R + r, live, other=0)
        z += correction
    maximum = tl.max(z, 0)
    chosen = tl.min(tl.where((z == maximum) & (token < V), token, V), 0)
    tl.store(VALUES + row * TILES + tile, maximum)
    tl.store(IDS + row * TILES + tile, chosen)


@triton.jit
def _argmax_merge(
    VALUES,
    IDS,
    OUT,
    V: tl.constexpr,
    K: tl.constexpr,
    TILES: tl.constexpr,
    BT: tl.constexpr,
    WEIGHTS=None,
):
    row = tl.program_id(0)
    if WEIGHTS is not None:
        if tl.load(WEIGHTS + row) <= 0:
            return
    tile = tl.arange(0, BT)
    z = tl.load(VALUES + row * TILES + tile, tile < TILES, other=-float("inf"))
    ids = tl.load(IDS + row * TILES + tile, tile < TILES, other=V)
    chosen = tl.min(tl.where(z == tl.max(z, 0), ids, V), 0)
    tl.store(OUT + row * K, chosen)


@torch.no_grad()
def greedy_propose(adapter, hidden, raw, workspace):
    """Candidate B: exactly the same FP32 correction, without top-k or exp."""
    n, vocab = raw.shape
    workspace.corrected_raw = None
    if n > workspace.rows or not raw.is_contiguous():
        raise ValueError("proposal workspace or contiguous logits contract violated")
    u = project_hidden(adapter, hidden, workspace)
    mode = {"auto": 3, "sparse": 1, "dense": 2}[adapter.config.proposal_mode]
    if not adapter.has_updated:
        mode = 0
    threshold = dispatch_threshold(adapter, n, hidden.shape[-1], vocab, hidden.dtype, raw.dtype)
    tiles = triton.cdiv(vocab, 1024)
    _argmax_scan[(n, tiles)](
        raw,
        u,
        adapter.B,
        adapter.active,
        adapter.active_count,
        workspace.maxima,
        workspace.argmax_ids,
        vocab,
        adapter.config.rank,
        tiles,
        mode,
        threshold,
        1024,
        DISPATCH_COUNTS=adapter.dispatch_counts,
        enable_fp_fusion=False,
    )
    _argmax_merge[(n,)](
        workspace.maxima,
        workspace.argmax_ids,
        workspace.ids,
        vocab,
        workspace.topk,
        tiles,
        triton.next_power_of_2(tiles),
    )
    return workspace.ids[:n], workspace.q[:n], workspace.norm[:n], u


def dispatch_threshold(adapter, n, hidden, vocab, dtype, raw_dtype=None):
    key = f"{adapter.device_name}|{n}|{hidden}|{vocab}|{adapter.config.rank}|{min(adapter.config.topk, vocab)}|{dtype}"
    if raw_dtype is not None and raw_dtype != dtype:
        key += f"|logits={raw_dtype}"
    entry = adapter.dispatch_cache.get("gpu_thresholds", {}).get(key)
    return int(entry["threshold_rows"]) if entry else vocab + 1


@torch.no_grad()
def statistics(adapter, u, raw, workspace, weights, raw_rows=None):
    """Reconstruct q for valid rows only, while B_t is still immutable.

    raw is the original compute_logits output retained until verification,
    never a second LM-head evaluation or a copy of B. Masked CTAs return
    before reading vocabulary logits. Launch shape stays static for graphs.
    """
    n, vocab = len(weights), raw.shape[-1]
    tiles, k, rank = triton.cdiv(vocab, 256), workspace.topk, adapter.config.rank
    mode = 1 if adapter.has_updated else 0
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
        WEIGHTS=weights,
        ROWS=raw_rows,
        enable_fp_fusion=False,
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
        WEIGHTS=weights,
        enable_fp_fusion=False,
    )


@triton.jit
def _teacher_flags(
    VALUES,
    IDS,
    WEIGHTS,
    FLAGS,
    SORTED,
    P,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
    BK: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, BK)
    v = tl.load(VALUES + row * WIDTH + j, j < K, other=-float("inf"))
    ids = tl.load(IDS + row * WIDTH + j, j < K, other=2147483647)
    flag = False
    higher = 0
    boundary = 0.0
    if WIDTH > K:
        boundary = tl.load(VALUES + row * WIDTH + K - 1)
        next_value = tl.load(VALUES + row * WIDTH + K)
        flag = (boundary > 0) & (boundary == next_value)
        higher = tl.sum((v > boundary).to(tl.int32), 0)
    tl.store(FLAGS + row, flag & (tl.load(WEIGHTS + row) > 0))
    for slot in range(K):
        best = tl.max(v, 0)
        chosen = tl.min(tl.where(v == best, ids, 2147483647), 0)
        tl.store(SORTED + row * K + slot, chosen)
        v = tl.where(ids == chosen, -float("inf"), v)
        ids = tl.where(ids == chosen, 2147483647, ids)

    if flag and tl.load(WEIGHTS + row) > 0:
        offset = tl.arange(0, BV)
        base, found = 0, 0
        while base < V and found < K - higher:
            token = base + offset
            value = tl.load(P + row * V + token, token < V, other=-float("inf"))
            match = (token < V) & (value == boundary)
            prefix = tl.cumsum(match.to(tl.int32))
            write_slot = higher + found + prefix - 1
            tl.store(SORTED + row * K + write_slot, token, match & (write_slot < K))
            found += tl.sum(match.to(tl.int32), 0)
            base += BV
