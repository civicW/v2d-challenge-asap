#!/usr/bin/env python3
"""Fetch every Git-LFS object for the challenge repo clone via the LFS batch
API (github.com is reachable from the A100; the media CDN and git-lfs are not).

Pointers carry the expected sha256, so every download is verified before the
file is replaced in place.
"""

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
LFS_URL = sys.argv[2] if len(sys.argv) > 2 else "https://github.com/nvidia-isaac/video_to_data.git/info/lfs/objects/batch"
BRANCH = sys.argv[3] if len(sys.argv) > 3 else "main"


def find_pointers():
    out = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or ".git" in p.parts:
            continue
        try:
            if p.stat().st_size > 1024:
                continue
            head = p.read_bytes()[:64]
            if not head.startswith(b"version https://git-lfs"):
                continue
            text = p.read_text()
            oid = size = None
            for line in text.splitlines():
                if line.startswith("oid sha256:"):
                    oid = line.split(":", 1)[1].strip()
                elif line.startswith("size "):
                    size = int(line.split()[1])
            if oid and size is not None:
                out.append((p, oid, size))
        except Exception:  # noqa: BLE001
            continue
    return out


def batch_hrefs(oids):
    body = json.dumps({
        "objects": [{"oid": o, "size": s} for o, s in oids],
        "operation": "download",
        "ref": {"name": f"refs/heads/{BRANCH}"},
    }).encode()
    req = urllib.request.Request(
        LFS_URL, data=body,
        headers={"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    return {o["oid"]: o.get("actions", {}).get("download", {}).get("href") for o in data["objects"]}


def fetch(path: Path, oid: str, href: str) -> str:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(href, timeout=300) as r:
                data = r.read()
            if hashlib.sha256(data).hexdigest() == oid:
                tmp = path.with_suffix(path.suffix + ".lfsdl")
                tmp.write_bytes(data)
                tmp.replace(path)
                return f"ok {path.name} {len(data)}"
            return f"HASHFAIL {path.name}"
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3)
    return f"FAIL {path.name}: {last}"


pointers = find_pointers()
print(f"{len(pointers)} LFS pointers found", flush=True)
done = 0
for i in range(0, len(pointers), 50):
    chunk = pointers[i:i + 50]
    try:
        hrefs = batch_hrefs([(oid, size) for _, oid, size in chunk])
    except Exception as e:  # noqa: BLE001
        print(f"batch error: {e}", flush=True)
        time.sleep(10)
        hrefs = batch_hrefs([(oid, size) for _, oid, size in chunk])
    for path, oid, size in chunk:
        href = hrefs.get(oid)
        if not href:
            print(f"NOHREF {path}")
            continue
        print(fetch(path, oid, href), flush=True)
        done += 1
print(f"ALL DONE {done}/{len(pointers)}", flush=True)
