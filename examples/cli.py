"""CLI: generate one pet bundle and write it to a .zip.

    python examples/cli.py "red panda" -o red_panda.zip
    python examples/cli.py "blue jay"     # a bird: walk=hop, run=fly
    python examples/cli.py "ball python"  # a crawler: walk/run = slow/fast slither

Every pet gets idle + walk + run + excited; the walk/run sprite motion is
tailored to the kind of animal. Requires ComfyUI running with the models
listed in the README.
"""
import argparse
import time
from pathlib import Path

from pet_factory import make_pet_zip


def main():
    ap = argparse.ArgumentParser(description="animal name -> DatsMe pet .zip")
    ap.add_argument("animal", help='e.g. "red panda", "penguin", "baby dragon"')
    ap.add_argument("-o", "--out", default=None, help="output .zip path")
    args = ap.parse_args()

    t0 = time.time()
    breed_id, data = make_pet_zip(
        args.animal,
        on_progress=lambda m, p: print(f"[{p*100:3.0f}%] {m}", flush=True))
    out = args.out or f"{breed_id}.zip"
    Path(out).write_bytes(data)
    print(f"wrote {out} ({len(data)} bytes) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
