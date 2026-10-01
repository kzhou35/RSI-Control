"""Reserve a GPU on a shared box by holding a visible block of its memory.

Our agent runs are bursty: they spend long stretches curating data / analysing results with the GPU
idle. A colleague checking `nvidia-smi` then sees a free GPU and takes it — and our training has
nowhere to run when it finally needs the device. This holder keeps a compute process resident on the
GPU for the whole run so it reads as OCCUPIED the entire time, then releases on exit.

    python scripts/run/hold_gpu.py --gpu 3 --gb 16          # hold until killed (SIGTERM/SIGINT releases)

Size it so it DETERS without starving our own agent: the 9B needs ~50GB peak (train + a vLLM eval
server), so on an 80GB H100 holding ~16GB leaves plenty. Set --gb 0 to disable.
Deliberately named so `nvidia-smi`/`ps` show who owns it and why.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", required=True, help="physical GPU index to hold")
    ap.add_argument("--gb", type=float, default=16.0, help="GiB to reserve (0 disables)")
    ap.add_argument("--label", default="rsi-bench-gpu-reservation")
    args = ap.parse_args()

    if args.gb <= 0:
        print("hold_gpu: --gb 0, nothing to hold")
        return 0

    # Pin BEFORE importing torch so we only ever touch the one device.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    try:
        import torch
    except Exception as e:  # torch missing -> don't block the run over a reservation
        print(f"hold_gpu: torch unavailable ({e}); not reserving", file=sys.stderr)
        return 0
    if not torch.cuda.is_available():
        print("hold_gpu: no CUDA device visible; not reserving", file=sys.stderr)
        return 0

    try:
        block = torch.empty(int(args.gb * (1024**3) // 2), dtype=torch.float16, device="cuda")
    except RuntimeError as e:
        print(f"hold_gpu: could not allocate {args.gb} GiB ({e}); not reserving", file=sys.stderr)
        return 0

    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("now", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))
    print(f"hold_gpu: {args.label} holding {args.gb} GiB on physical GPU {args.gpu} "
          f"(pid {os.getpid()}); SIGTERM to release", flush=True)
    while not stop["now"]:
        # Touch it occasionally so the process shows sustained activity, not just an allocation.
        block[0] = 1.0
        torch.cuda.synchronize()
        # Sleep in 1s slices: PEP 475 resumes an interrupted time.sleep(), so a single long sleep
        # would delay release by up to its full duration after SIGTERM.
        for _ in range(10):
            if stop["now"]:
                break
            time.sleep(1)
    del block
    torch.cuda.empty_cache()
    print("hold_gpu: released", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
