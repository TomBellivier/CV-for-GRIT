import os, threading, time
import psutil
from insectpose.registry import load_all_plugins; load_all_plugins()
from insectpose.cli import load_config
from insectpose import pipeline

proc = psutil.Process(os.getpid())
peak = {"value": 0.0}

def tree_rss_gb():
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total / 2**30

def monitor():
    while True:
        peak["value"] = max(peak["value"], tree_rss_gb())
        time.sleep(0.5)

threading.Thread(target=monitor, daemon=True).start()

def mark(phase):
    print(f"== {phase:14s} | RSS arbre {tree_rss_gb():6.2f} Go | pic {peak['value']:6.2f} Go",
          flush=True)
    peak["value"] = 0.0

cfg = load_config(["experiment=exp_a_yolo_pooled", "fold=0", "train.epochs=2"])
ctx, data, approach = pipeline._prepare_run(cfg)
ctx.setup()
mark("prepare")
approach.fit(data, ctx);                          mark("fit")
approach.predict(data.role("val"), ctx, "val");   mark("predict val")
approach.predict(data.role("test"), ctx, "test"); mark("predict test")
print("== OK ==")