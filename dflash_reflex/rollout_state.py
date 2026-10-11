"""Stable request identities; storage is independent of vLLM batch ordering."""

from dataclasses import dataclass

import torch


@dataclass
class RequestSlot:
    index: int
    generation: int
    pending: bool


class CohortState:
    @torch.inference_mode(False)
    def __init__(self, group_id, capacity, width, hidden, rank, topk, device, dtype):
        self.group_id = str(group_id)
        self.capacity, self.width = capacity, width
        self.requests = {}
        self.free = list(reversed(range(capacity)))
        self.h = torch.empty(capacity, width, hidden, device=device, dtype=dtype)
        self.u = torch.empty(capacity, width, rank, device=device, dtype=torch.float32)
        self.ids = torch.empty(capacity, width, topk, device=device, dtype=torch.long)
        self.q = torch.empty(capacity, width, topk, device=device, dtype=torch.float32)
        self.norm = torch.empty(capacity, width, 2, device=device, dtype=torch.float32)

    def proposal_slots(self, request_ids):
        result = []
        for request_id in request_ids:
            slot = self.requests.get(request_id)
            if slot is None:
                if not self.free:
                    raise RuntimeError("cohort request capacity exhausted")
                slot = self.requests[request_id] = RequestSlot(self.free.pop(), 0, False)
            # A preempted/unverified proposal can be superseded by a new one.
            slot.generation += 1
            slot.pending = True
            result.append(slot.index)
        return result

    def verification_slots(self, request_ids, lengths):
        result = []
        for request_id, length in zip(request_ids, lengths, strict=True):
            slot = self.requests.get(request_id)
            if length:
                if slot is None or not slot.pending:
                    raise RuntimeError(f"verification without proposal: {request_id}")
                slot.pending = False
                result.append(slot.index)
            else:
                result.append(-1)
        return result

    def finish_request(self, request_id):
        slot = self.requests.pop(request_id, None)
        if slot is not None:
            self.free.append(slot.index)


class CohortRegistry:
    """CPU lifecycle oracle; ending a response never resets its cohort's B."""

    def __init__(self):
        self.groups = {}

    def begin(self, group_id, factory):
        if group_id in self.groups:
            raise RuntimeError("duplicate live cohort")
        self.groups[group_id] = factory()
        return self.groups[group_id]

    def end(self, group_id):
        state = self.groups.pop(group_id)
        state.reset()
