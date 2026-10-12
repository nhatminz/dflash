"""Reusable host metadata and GPU extraction for synchronous DFlash blocks."""

import torch
import triton
import triton.language as tl


@triton.jit
def _limits_and_counts(
    M,
    OUT,
    ACCEPT,
    DISCARD,
    LIMITS,
    COUNTERS,
    COLS: tl.constexpr,
    OW: tl.constexpr,
    BO: tl.constexpr,
    STOP: tl.constexpr,
    BS: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    AS: tl.constexpr,
    DS: tl.constexpr,
):
    req = tl.program_id(0)
    length = tl.load(M + req * COLS)
    limit = tl.load(M + req * COLS + 3)
    pos = tl.arange(0, BO)
    output = tl.load(OUT + req * OS0 + pos * OS1, pos < OW, other=-1)
    if STOP:
        slot = tl.arange(0, BS)
        stops = tl.load(M + req * COLS + 4 + slot, slot < STOP, other=-2)
        eos = (
            tl.sum(((output[:, None] == stops[None, :]) & (slot[None, :] < STOP)).to(tl.int32), 1)
            > 0
        )
        first = tl.min(tl.where(eos & (pos < OW), pos, OW), 0)
        limit = tl.minimum(limit, first + 1)
    tl.store(LIMITS + req, limit)
    live = tl.load(DISCARD + req * DS) == 0
    count = tl.sum(((output >= 0) & (pos < OW) & (pos < limit) & live).to(tl.int32), 0)
    verified = (length > 0) & live
    accepted = tl.minimum(tl.load(ACCEPT + req * AS), limit) * live
    proposed = tl.minimum(length, limit) * live
    tl.atomic_add(COUNTERS, tl.where(verified, count, 0).to(tl.float64))
    tl.atomic_add(COUNTERS + 1, verified.to(tl.float64))
    tl.atomic_add(COUNTERS + 2, tl.where(verified, accepted, 0).to(tl.float64))
    tl.atomic_add(COUNTERS + 3, proposed.to(tl.float64))


@triton.jit
def _extract(
    M,
    LIMITS,
    ACCEPT,
    DISCARD,
    TOKENS,
    H,
    U,
    DI,
    DQ,
    NORM,
    OH,
    OU,
    ODI,
    ODQ,
    ONORM,
    WEIGHTS,
    RAW_ROWS,
    BATCH: tl.constexpr,
    COLS: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    BB: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
    WA: tl.constexpr,
    WR: tl.constexpr,
    FUSED: tl.constexpr,
    AS: tl.constexpr,
    DS: tl.constexpr,
    TS: tl.constexpr,
):
    row = tl.program_id(0)
    reqs = tl.arange(0, BB)
    starts = tl.load(M + reqs * COLS + 1, reqs < BATCH, other=2147483647)
    req = tl.sum(((reqs < BATCH) & (row >= starts)).to(tl.int32), 0) - 1
    position = row - tl.load(M + req * COLS + 1)
    source = tl.load(M + req * COLS + 2) + position
    accepted = tl.load(ACCEPT + req * AS)
    valid = (
        (position <= accepted)
        & (position < tl.load(LIMITS + req))
        & (tl.load(DISCARD + req * DS) == 0)
        & (tl.load(TOKENS + row * TS) >= 0)
    )
    weight = tl.where(valid, tl.where(position < accepted, WA, WR), 0.0)
    active = weight > 0
    d, r, k = tl.arange(0, BD), tl.arange(0, BR), tl.arange(0, BK)
    h = tl.load(H + source * D + d, active & (d < D), other=0).to(tl.float32)
    u = tl.load(U + source * R + r, active & (r < R), other=0)
    tl.store(OH + row * D + d, h, d < D)
    tl.store(OU + row * R + r, u, r < R)
    if FUSED:
        ids = tl.load(DI + source * K + k, active & (k < K), other=0)
        q = tl.load(DQ + source * K + k, active & (k < K), other=0)
        tl.store(ODI + row * K + k, ids, k < K)
        tl.store(ODQ + row * K + k, q, k < K)
        tl.store(ONORM + row * 2, tl.load(NORM + source * 2, active, other=0))
        tl.store(ONORM + row * 2 + 1, tl.load(NORM + source * 2 + 1, active, other=1))
    tl.store(WEIGHTS + row, weight)
    tl.store(RAW_ROWS + row, source)


class MetadataWorkspace:
    @torch.inference_mode(False)
    def __init__(self, capacity, width, hidden, rank, topk, device):
        self.capacity, self.device = capacity, device
        rows = capacity * width
        self.h = torch.empty(rows, hidden, device=device, dtype=torch.float32)
        self.u = torch.empty(rows, rank, device=device, dtype=torch.float32)
        self.ids = torch.empty(rows, topk, device=device, dtype=torch.long)
        self.q = torch.empty(rows, topk, device=device)
        self.norm = torch.empty(rows, 2, device=device)
        self.weights = torch.empty(rows, device=device)
        self.raw_rows = torch.empty(rows, device=device, dtype=torch.long)
        self.limits = torch.empty(capacity, device=device, dtype=torch.long)
        self.staging = []
        self.cursor = 0

    def upload(self, lengths, limits, eos, raw_starts):
        stop = max(map(len, eos), default=0)
        cols = triton.next_power_of_2(4 + stop)
        index = self.cursor % max(len(self.staging), 1)
        stage = self.staging[index] if self.staging else None
        if stage is None or stage[0].shape[1] < cols or not stage[2].query():
            # Never mutate pinned host storage while an asynchronous H2D or
            # extraction is still using it. Nonblocking query; no host wait.
            host = torch.empty(self.capacity, cols, dtype=torch.long, pin_memory=True)
            gpu = torch.empty(self.capacity, cols, device=self.device, dtype=torch.long)
            stage = (host, gpu, torch.cuda.Event())
            if index < len(self.staging) and self.staging[index][2].query():
                self.staging[index] = stage
            else:
                self.staging.append(stage)
                index = len(self.staging) - 1
        host, gpu, done = stage
        array = host.numpy()
        array[: len(lengths)].fill(-2)
        start = 0
        for i, (length, limit, stops, raw) in enumerate(
            zip(lengths, limits, eos, raw_starts, strict=True)
        ):
            array[i, :4] = length, start, raw, limit
            array[i, 4 : 4 + len(stops)] = stops
            start += length
        gpu[: len(lengths)].copy_(host[: len(lengths)], non_blocking=True)
        self.cursor = index + 1
        return gpu, stop, done

    def extract(self, runtime, accepted, metadata, output):
        lengths = metadata.num_draft_tokens
        raw_starts = [runtime.raw_requests.get(key, 0) for key in runtime.request_ids]
        matrix, stop, done = self.upload(lengths, runtime.remaining, runtime.eos, raw_starts)
        batch, cols = len(lengths), matrix.shape[1]
        _limits_and_counts[(batch,)](
            matrix,
            output,
            accepted,
            runtime.discard,
            self.limits,
            runtime.counters,
            cols,
            output.shape[1],
            triton.next_power_of_2(output.shape[1]),
            stop,
            triton.next_power_of_2(max(stop, 1)),
            output.stride(0),
            output.stride(1),
            accepted.stride(0),
            runtime.discard.stride(0),
        )
        n = metadata.draft_token_ids.numel()
        cache = runtime.cache
        _extract[(n,)](
            matrix,
            self.limits,
            accepted,
            runtime.discard,
            metadata.draft_token_ids,
            cache.h,
            cache.u,
            cache.ids,
            cache.q,
            cache.norm,
            self.h,
            self.u,
            self.ids,
            self.q,
            self.norm,
            self.weights,
            self.raw_rows,
            batch,
            cols,
            self.h.shape[1],
            self.u.shape[1],
            self.ids.shape[1],
            triton.next_power_of_2(batch),
            triton.next_power_of_2(self.h.shape[1]),
            triton.next_power_of_2(self.u.shape[1]),
            triton.next_power_of_2(self.ids.shape[1]),
            runtime.config.accepted_weight,
            runtime.config.rejected_weight,
            runtime.pipeline == "fused",
            accepted.stride(0),
            runtime.discard.stride(0),
            metadata.draft_token_ids.stride(0),
        )
        done.record()
        return (
            self.h[:n],
            self.u[:n],
            self.ids[:n],
            self.q[:n],
            self.norm[:n],
            self.weights[:n],
            self.raw_rows[:n],
        )
