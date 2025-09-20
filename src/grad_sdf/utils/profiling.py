import time

import torch
from tqdm import tqdm


class CpuTimer:

    def __init__(self, message, repeats: int = 1, warmup: int = 0):
        self.message = message
        self.repeats = repeats
        self.warmup = warmup
        self.cnt = 0
        self.t = 0
        self.average_t = 0
        self._total_t = 0
        self.total_t = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        if self.cnt < self.warmup:
            self.cnt += 1
            return
        self.cnt += 1
        assert self.cnt <= self.repeats
        self.t = self.end - self.start
        self._total_t += self.t
        self.average_t = self._total_t / (self.cnt - self.warmup)
        self.total_t = self.average_t * self.cnt
        tqdm.write(f"{self.message}: {self.t:.6f}(cur)/{self.average_t:.6f}(avg)/{self.total_t:.6f}(total) seconds")


def cpu_timer(message, repeats, warmup):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with CpuTimer(message, repeats, warmup):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class GpuTimer:

    def __init__(self, message, repeats: int = 1, warmup: int = 0, enable: bool = True):
        self.message = message
        self.repeats = repeats
        self.warmup = warmup
        self.enable = enable
        self.cnt = 0
        self.t = 0
        self.average_t = 0
        self._total_t = 0
        self.total_t = 0

    def __enter__(self):
        if not self.enable:
            return self
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, *args):
        if not self.enable:
            return
        self.end.record()
        torch.cuda.synchronize()
        self.t = self.start.elapsed_time(self.end) / 1e3
        if self.cnt < self.warmup:
            self.cnt += 1
            return
        self.cnt += 1
        assert self.cnt <= self.repeats
        self._total_t += self.t
        self.average_t = self._total_t / (self.cnt - self.warmup)
        self.total_t = self.average_t * self.cnt
        tqdm.write(f"{self.message}: {self.t:.6f}(cur)/{self.average_t:.6f}(avg)/{self.total_t:.6f}(total) seconds")


def gpu_timer(message, repeats, warmup, enable=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with GpuTimer(message, repeats, warmup, enable):
                return func(*args, **kwargs)

        return wrapper

    return decorator
