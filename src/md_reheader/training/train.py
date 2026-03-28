TARGET_EFFECTIVE_BS = 32


def compute_grad_accum_steps(bucket_batch_size: int, num_gpus: int) -> int:
    per_step = bucket_batch_size * num_gpus
    return max(1, TARGET_EFFECTIVE_BS // per_step)
