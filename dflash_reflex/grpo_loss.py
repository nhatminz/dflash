"""SpecNaacl's k3 KL gradient and sequence reduction, independent of backend."""

import torch


def specnaacl_kl_terms(logps, reference):
    delta = reference.detach() - logps
    return delta.exp() - delta - 1


def sequence_objective(logps, reference, advantages, mask, beta, epsilon, old=None):
    old = logps.detach() if old is None else old.detach()
    ratio = (logps - old).exp()
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon)
    actor = torch.minimum(ratio * advantages, clipped * advantages)
    terms = -actor + beta * specnaacl_kl_terms(logps, reference)
    count = mask.sum(-1)
    if (count <= 0).any():
        raise ValueError("sequence objective requires positive response-token counts")
    return ((terms * mask).sum(-1) / count).mean()
