import math

def linear_warmup(warmup_len):
    def wrapper(it):
        if it >= warmup_len or warmup_len == 0:
            return 1.0
        else:
            return (it+1) / (warmup_len+1)

    return wrapper


def linear_down(ramp_down_start, ramp_down_len, last_lr_value):
    def wrapper(it):
        if it < ramp_down_start:
            return 1.0
        elif it >= ramp_down_start + ramp_down_len:
            return last_lr_value
        else:
            return last_lr_value + (1.0 - last_lr_value) * (
                1 - (it - ramp_down_start) / ramp_down_len
            )

    return wrapper

def cosine_down(ramp_down_start, ramp_down_len, last_lr_value):
    def wrapper(it):
        if it < ramp_down_start:
            return 1.0
        elif it >= ramp_down_start + ramp_down_len:
            return last_lr_value
        else:
            return last_lr_value + (1.0 - last_lr_value) * (0.5*(1+math.cos(
                math.pi*(it - ramp_down_start) / ramp_down_len
            )))

    return wrapper


def linear_warmup_linear_down(
    warm_up_len, ramp_down_start, ramp_down_len, last_lr_value
):
    def wrapper(it):
        warmup = linear_warmup(warm_up_len)
        ramp_down = linear_down(ramp_down_start, ramp_down_len, last_lr_value)
        return warmup(it) * ramp_down(it)

    return wrapper

def linear_warmup_cosine_down(
    warm_up_len, ramp_down_start, ramp_down_len, last_lr_value
):
    def wrapper(it):
        warmup = linear_warmup(warm_up_len)
        ramp_down = cosine_down(ramp_down_start, ramp_down_len, last_lr_value)
        return warmup(it) * ramp_down(it)

    return wrapper