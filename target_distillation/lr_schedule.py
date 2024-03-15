from padertorch.train.hooks import LRAnnealingHook

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


def linear_warmup_linear_down(
    warm_up_len, ramp_down_start, ramp_down_len, last_lr_value
):
    def wrapper(it):
        warmup = linear_warmup(warm_up_len)
        ramp_down = linear_down(ramp_down_start, ramp_down_len, last_lr_value)
        return warmup(it) * ramp_down(it)

    return wrapper

def apply_lr_schedule(
        trainer,
        lr,
        warm_up_len=0,
        ramp_down_start=0,
        ramp_down_len=0,
        last_lr_value=1,
        num_iterations=1000,
        warmup_mode="lin_lin",
        interval=100,
    ):
    warm_up_len = int(num_iterations * warm_up_len)
    ramp_down_start = int(num_iterations * ramp_down_start)
    ramp_down_len = int(num_iterations * ramp_down_len)

    assert warmup_mode == "lin_lin", warmup_mode
    sched_fn = linear_warmup_linear_down(
        warm_up_len, ramp_down_start, ramp_down_len, last_lr_value
    )
    
    breakpoints = [
        (i, sched_fn(i))
        for i in range(
            0, num_iterations, interval
        )  # piece-wise linear interpolation
    ]
    lr_hook = LRAnnealingHook(
        trigger=(
            interval,
            "iteration",
        ),
        breakpoints=breakpoints,
        unit="iteration",
    )
    trainer.register_hook(lr_hook)
    lr_hook.scale = lr

def get_total_iterations(epochs, iterations, num_train_samples):
    if epochs is not None:
        return epochs * num_train_samples
    else:
        return iterations