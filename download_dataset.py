"""Download only the 12 single-condition COPE maps from ds004044 (public S3).
Resumable + parallel. No AWS CLI needed.
"""
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ENDPOINT = "https://s3.amazonaws.com/openneuro.org/"
PREFIX = "ds004044/derivatives/ciftify/"
DEST = Path("data/raw")
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

CONDS = {"Toe", "Ankle", "LeftLeg", "RightLeg", "Finger", "Wrist",
         "Forearm", "Upperarm", "Jaw", "Lip", "Tongue", "Eye"}


def list_keys(prefix):
    token = None
    while True:
        qs = urllib.parse.urlencode({"list-type": 2, "prefix": prefix, "max-keys": 500}
                                    | ({"continuation-token": token} if token else {}))
        with urllib.request.urlopen(f"{ENDPOINT}?{qs}", timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.iter(f"{NS}Contents"):
            yield c.findtext(f"{NS}Key"), int(c.findtext(f"{NS}Size") or 0)
        if root.findtext(f"{NS}IsTruncated") != "true":
            break
        token = root.findtext(f"{NS}NextContinuationToken")


def want(key):
    if "results/ses-1_task-motor_hp200_s4_level2.feat/" not in key:
        return False
    if not key.endswith(".dscalar.nii"):
        return False
    if "level2_cope_" not in key:
        return False
    seg = key.split("level2_cope_")[1].split("_hp200_s4")[0]
    return seg in CONDS


def fetch(job):
    key, size = job
    tgt = DEST / key
    if tgt.exists() and tgt.stat().st_size == size:
        return "skip"
    tgt.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(ENDPOINT + urllib.parse.quote(key), tgt)
    return "ok"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    jobs = [(k, s) for k, s in list_keys(PREFIX) if want(k)]
    total = sum(s for _, s in jobs)
    print(f"Found {len(jobs)} COPE files, {total/1e9:.2f} GB", file=sys.stderr)
    t0 = time.time()
    done = ok = skip = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(fetch, j) for j in jobs]
        for f in as_completed(futs):
            done += 1
            if f.result() == "ok":
                ok += 1
            else:
                skip += 1
            if done % 50 == 0:
                rate = (done) / (time.time() - t0)
                print(f"  {done}/{len(jobs)}  ({ok} new, {skip} skipped)  {rate:.1f} files/s",
                      file=sys.stderr)
    print(f"DONE. new={ok} skipped={skip}", file=sys.stderr)


if __name__ == "__main__":
    main()


