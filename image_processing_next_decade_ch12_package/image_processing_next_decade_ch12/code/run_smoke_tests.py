from __future__ import annotations

import contextlib
import io
import runpy
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "vae_minimal.py", "vq_vae_minimal.py", "gan_minimal.py",
    "wgan_gp_minimal.py", "realnvp_2d.py", "pixelcnn_mask.py",
    "fid_kid_toy.py", "latent_interpolation.py", "flow_matching_2d.py",
]

def main() -> None:
    blocks: list[str] = []
    for name in SCRIPTS:
        stream = io.StringIO()
        start = time.perf_counter()
        status = 0
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                runpy.run_path(str(ROOT / name), run_name="__main__")
        except Exception as exc:  # show the exact failing script
            status = 1
            stream.write(f"{type(exc).__name__}: {exc}\n")
        elapsed = time.perf_counter() - start
        blocks.append(f"## {name}\nseconds={elapsed:.3f}\n{stream.getvalue().rstrip()}\nexit={status}\n")
        if status:
            raise RuntimeError(f"Smoke test failed: {name}")
    output = "\n".join(blocks) + "\n"
    out_path = ROOT.parent / "outputs" / "smoke_test.txt"
    out_path.write_text(output, encoding="utf-8")
    print(output)

if __name__ == "__main__":
    main()
