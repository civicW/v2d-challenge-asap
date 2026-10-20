import json, os, sys, time, urllib.request
from pathlib import Path
BASE = "https://hf-mirror.com/api/datasets/nvidia/video_to_data_challenge"
RAW = "https://hf-mirror.com/datasets/nvidia/video_to_data_challenge/resolve/main"
ROOT = Path.home() / "work/v2d/dataset"
HDR = {"User-Agent": "Mozilla/5.0 (fetch-track3)"}

def get(url, timeout=90):
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def api(path, retries=5):
    for i in range(retries):
        try:
            return json.loads(get(f"{BASE}/tree/main/{path}"))
        except Exception as e:
            print("api retry", path, e, flush=True)
            time.sleep(8 * (i + 1))
    return None

def fetch(sub):
    entries = api(sub)
    if entries is None:
        print("SKIP-DIR", sub, flush=True); return
    for entry in entries:
        p = entry["path"]
        if entry["type"] == "directory":
            fetch(p); continue
        dest = ROOT / p
        if dest.exists() and dest.stat().st_size == entry.get("size", -1):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        for i in range(4):
            try:
                data = get(f"{RAW}/{p}")
                tmp.write_bytes(data)
                os.replace(tmp, dest)
                print("ok", p, len(data), flush=True)
                break
            except Exception as e:
                print("retry", p, e, flush=True)
                time.sleep(10 * (i + 1))
        else:
            print("FAIL", p, flush=True)

for sub in sys.argv[1:]:
    print("== fetching", sub, flush=True)
    fetch(sub)
print("ALL DONE", flush=True)
