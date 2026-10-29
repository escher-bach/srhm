"""Smoke test for the Colab L4 pipeline: prints one JSON line and exits."""
import json, os, subprocess, sys, time
t0 = time.time()
info = {"budget_seconds": int(os.environ.get("BUDGET_SECONDS", "0"))}
try:
    import torch
    info.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    if torch.cuda.is_available():
        x = torch.randn(4096, 4096, device="cuda"); torch.cuda.synchronize()
        t = time.time(); y = x @ x; torch.cuda.synchronize(); info["matmul_ms"] = round(1000 * (time.time() - t), 1)
except Exception as e:
    info["error"] = repr(e)
info["disk_free_gb"] = round(os.statvfs("/content").f_bavail * os.statvfs("/content").f_frsize / 1e9, 1)
info["secs"] = round(time.time() - t0, 1)
print(json.dumps(info))
if info.get("gpu") is None or "L4" not in info["gpu"]:
    sys.exit(3)          # not an L4: fail loudly, the wrapper still tears down
