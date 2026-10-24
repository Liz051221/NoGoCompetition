"""Memory-bounded backward for one unchanged effective weighted batch.

The caller creates its permutation and augmentation once per effective batch,
zeros gradients once, and performs exactly one optimizer/scaler step afterward.
BatchNorm running statistics must be frozen by the caller.
"""
from __future__ import annotations


def backward_microbatches(
    torch, model, teacher, scaler, x, target_policy, target_value,
    policy_sample_weight, value_sample_weight, *, micro_batch_size,
    policy_weight, value_weight, distill_policy_weight, distill_value_weight,
    use_amp,
):
    if micro_batch_size < 1 or len(x) < 1:
        raise ValueError("micro batch size and effective batch must be positive")
    if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training
           for module in model.modules()):
        raise ValueError("micro batches require frozen BatchNorm statistics")
    if teacher is not None and teacher.training:
        raise ValueError("microbatch teacher must remain in eval mode")
    policy_denominator = policy_sample_weight.sum()
    value_denominator = value_sample_weight.sum()
    policy_active = bool(policy_denominator > 0)
    value_active = bool(value_denominator > 0)
    # These denominators span the EFFECTIVE batch, not each physical batch.
    # In particular, a physical batch can contain only masked new-value rows.
    totals = x.new_zeros(3)
    for start in range(0, len(x), micro_batch_size):
        stop = start + micro_batch_size
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            inputs = x[start:stop]
            logits, predicted = model(inputs)
            legal = inputs[:, 3].reshape(len(inputs), -1).bool()
            log_probs = torch.log_softmax(logits.masked_fill(~legal, -1.0e4), dim=1)
            policy_per_sample = -(target_policy[start:stop] * log_probs).sum(dim=1)
            policy_term = ((policy_per_sample * policy_sample_weight[start:stop]).sum()
                           / policy_denominator if policy_active else logits.sum() * 0)
            value_term = (((predicted - target_value[start:stop]).square()
                           * value_sample_weight[start:stop]).sum()
                          / value_denominator if value_active else predicted.sum() * 0)
            distill_policy_term = logits.sum() * 0
            distill_value_term = predicted.sum() * 0
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits, teacher_value = teacher(inputs)
                    teacher_probs = torch.log_softmax(
                        teacher_logits.masked_fill(~legal, -1.0e4), dim=1).exp()
                if distill_policy_weight > 0:
                    distill_policy_term = torch.nn.functional.kl_div(
                        log_probs, teacher_probs, reduction="none").sum() / len(x)
                if distill_value_weight > 0:
                    distill_value_term = (predicted - teacher_value).square().sum() / len(x)
            loss = (policy_weight * policy_term + value_weight * value_term
                    + distill_policy_weight * distill_policy_term
                    + distill_value_weight * distill_value_term)
        scaler.scale(loss).backward()
        totals += torch.stack((loss.detach(), policy_term.detach(), value_term.detach()))
    return tuple(totals.unbind())
