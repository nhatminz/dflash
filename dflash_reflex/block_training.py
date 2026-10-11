"""CPU contracts used to independently test NVIDIA's block alignment."""

import math

import torch


def construct_blocks(input_ids, anchors, block_size, mask_id):
    offsets = torch.arange(block_size, device=input_ids.device)
    positions = anchors[..., None] + offsets
    blocks = torch.full_like(positions, mask_id)
    blocks[..., 0] = input_ids.gather(1, anchors)
    targets = input_ids.gather(1, positions[..., 1:].clamp(max=input_ids.shape[1] - 1).flatten(1))
    return blocks, positions, targets.view(*anchors.shape, block_size - 1)


def block_attention_mask(anchors, seq_len, block_size, attention_mask=None):
    b, n = anchors.shape
    query = n * block_size
    trunk = torch.arange(seq_len, device=anchors.device)[None, None, :]
    visible = trunk < anchors.repeat_interleave(block_size, dim=1)[..., None]
    if attention_mask is not None:
        visible &= attention_mask[:, None, :].bool()
    block_ids = torch.arange(query, device=anchors.device) // block_size
    own = (block_ids[:, None] == block_ids[None, :]).expand(b, -1, -1)
    return torch.cat((visible, own), -1)[:, None]


def slot_mask(token_mask, sample_mask, anchors, anchor_valid, gamma):
    positions = anchors[..., None] + 1 + torch.arange(gamma, device=anchors.device)
    labels = token_mask.gather(1, positions.clamp(max=token_mask.shape[1] - 1).flatten(1)).view_as(
        positions
    )
    live = (
        (positions < token_mask.shape[1])
        & labels.bool()
        & anchor_valid[..., None]
        & sample_mask[:, None, None].bool()
    )
    return live.int().cumprod(-1).bool()


def slot_weights(block_size):
    points = [(8, 4.0), (10, 5.0), (16, 7.0)]
    lo, hi = points[:2] if block_size <= 10 else points[1:]
    decay = max(1.0, lo[1] + (hi[1] - lo[1]) * (block_size - lo[0]) / (hi[0] - lo[0]))
    return [math.exp(-j / decay) for j in range(block_size - 1)]


def block_loss_reference(student, teacher, anchors, mask, weights):
    gamma = student.shape[2]
    position = (anchors[..., None] + torch.arange(gamma, device=anchors.device)).clamp(
        max=teacher.shape[1] - 1
    )
    gathered = teacher.gather(1, position.flatten(1)[..., None].expand(-1, -1, teacher.shape[-1]))
    probabilities = gathered.detach().float().softmax(-1).view_as(student)
    ce = -(probabilities * student.float().log_softmax(-1)).sum(-1)
    w = torch.as_tensor(weights, device=student.device) * mask
    return (ce * w).sum() / w.sum().clamp_min(1)
