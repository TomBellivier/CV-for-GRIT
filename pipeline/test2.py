# diag2.py
import linecache, os, threading, time, tracemalloc
from insectpose.registry import load_all_plugins; load_all_plugins()
from insectpose.cli import load_config
from insectpose import pipeline
from insectpose.evaluation.evaluator import evaluate_run

def rss_gb():
    with open("/proc/self/status") as f:          # processus principal seul
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1048576

peak = {"v": 0.0}
def monitor():
    while True:
        peak["v"] = max(peak["v"], rss_gb()); time.sleep(0.5)
threading.Thread(target=monitor, daemon=True).start()

def mark(phase):
    print(f"== {phase:16s} RSS {rss_gb():6.2f} Go | pic {peak['v']:6.2f} Go", flush=True)
    peak["v"] = 0.0
    snap = tracemalloc.take_snapshot()
    for stat in snap.statistics("lineno")[:5]:
        frame = stat.traceback[0]
        print(f"     {stat.size/2**20:8.1f} Mo  {frame.filename.split('/')[-1]}:{frame.lineno}"
              f"  {linecache.getline(frame.filename, frame.lineno).strip()[:70]}", flush=True)

tracemalloc.start(1)
cfg = load_config(["experiment=exp_a_yolo_pooled", "fold=0", "train.epochs=2"])
ctx, data, approach = pipeline._prepare_run(cfg); ctx.setup()
mark("prepare")
approach.fit(data, ctx);                          mark("fit")
approach.predict(data.role("val"), ctx, "val");   mark("predict val")
approach.predict(data.role("test"), ctx, "test"); mark("predict test")
annotations, schemas = pipeline._load_context_data(cfg, ctx.paths)
evaluate_run(ctx.run_id, ctx.paths, annotations, schemas, cfg.eval, approach="yolo_pooled")
mark("evaluate")
print("== OK ==")