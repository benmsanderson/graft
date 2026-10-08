"""Fetch LUH3 files from ESGF, from every mirror at once, checksum-verified.

ESGF serves LUH3 at roughly 0.1-1 MB/s per connection on every node, and
nodes drop long transfers: a plain download of the 6 GB historic states file
takes most of a day and may not finish. The same files are replicated at
several nodes, all support byte ranges, and ESGF publishes a SHA-256 for
each. So this asks the federation's indexes for every replica, and hands
aria2 all of them per file with the checksum, which splits the transfer into
segments across mirrors and connections, retries dropped ones, and refuses a
file that does not verify. About 2-4 MB/s here, against 0.1 from one node.

    python scripts/fetch_luh3.py ~/madrat/sources/LUH3
    python scripts/fetch_luh3.py ~/madrat/sources/LUH3 --kinds states management static
    python scripts/fetch_luh3.py DIR --source-id UofMD-landState-vl-3-1-1 --kinds states

Without --run it writes the aria2 input file and prints the command; with it,
it runs aria2c (brew install aria2). Everything it reads is public, and it
needs nothing beyond the standard library.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# Indexes that between them see the European replicas and ORNL. Each is
# asked independently; one being down does not stop the others.
INDEXES = [
    "https://esgf-data.dkrz.de/esg-search/search",
    "https://esgf.ceda.ac.uk/esg-search/search",
    "https://esgf-node.ornl.gov/esgf-1-5-bridge",
]

KINDS = {
    "states": "multiple-states",
    "management": "multiple-management",
    "transitions": "multiple-transitions",
    "static": "multiple-static",
}


def query(index: str, source_id: str) -> list[dict]:
    url = (f"{index}?type=File&project=input4MIPs&source_id={source_id}"
           "&limit=200&format=application%2Fsolr%2Bjson")
    try:
        with urllib.request.urlopen(url, timeout=240) as r:
            return json.load(r)["response"]["docs"]
    except Exception as error:  # an unreachable index is not fatal
        print(f"  {index}: {type(error).__name__}, skipped", file=sys.stderr)
        return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dest", help="directory to download into, e.g. ~/madrat/sources/LUH3")
    ap.add_argument("--source-id", default="UofMD-landState-3-1-1",
                    help="input4MIPs source_id (default: LUH3 historic)")
    ap.add_argument("--kinds", nargs="+", default=["states", "management", "static"],
                    choices=sorted(KINDS),
                    help="which files (default: states management static; transitions is 17 GB)")
    ap.add_argument("--run", action="store_true", help="run aria2c instead of printing the command")
    args = ap.parse_args(argv)

    wanted = {KINDS[k] for k in args.kinds}
    files: dict[str, dict] = {}
    for index in INDEXES:
        for doc in query(index, args.source_id):
            title = doc.get("title", "")
            if not any(title.startswith(w + "_") for w in wanted):
                continue
            entry = files.setdefault(title, {"size": doc.get("size"), "urls": [], "sha256": None})
            if doc.get("checksum_type") and doc["checksum_type"][0].upper() == "SHA256":
                entry["sha256"] = entry["sha256"] or doc["checksum"][0]
            for u in doc.get("url", []):
                link = u.split("|")[0]
                # THREDDS file servers, and Globus HTTPS endpoints (newer datasets may be served only there)
                direct = "/fileServer/" in link or (link.startswith("https://") and ".data.globus.org/" in link)
                if direct and link not in entry["urls"]:
                    entry["urls"].append(link)

    if not files:
        print(f"no files found for {args.source_id} {args.kinds}", file=sys.stderr)
        return 1

    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    listing = dest / ".luh3.aria2"
    with listing.open("w") as f:
        for title, entry in sorted(files.items()):
            f.write("\t".join(entry["urls"]) + "\n")
            f.write(f"  out={title}\n")
            if entry["sha256"]:
                f.write(f"  checksum=sha-256={entry['sha256']}\n")

    total = 0
    for title, entry in sorted(files.items()):
        size = entry["size"] or 0
        total += size
        hosts = ", ".join(u.split("/")[2] for u in entry["urls"])
        check = "sha256" if entry["sha256"] else "NO CHECKSUM"
        print(f"{title}\n  {size / 1e9:.2f} GB, {len(entry['urls'])} mirrors ({hosts}), {check}")
    print(f"\n{total / 1e9:.2f} GB in {len(files)} files")

    command = ["aria2c", "-i", str(listing), "-d", str(dest), "-j", "2", "-x", "8", "-s", "24",
               "-k", "16M", "--file-allocation=none", "--max-tries=0", "--retry-wait=15",
               "--timeout=60", "--summary-interval=60", "--console-log-level=warn"]
    if not args.run:
        print("\n" + " ".join(command))
        return 0
    if shutil.which("aria2c") is None:
        print("aria2c not found; brew install aria2", file=sys.stderr)
        return 1
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
