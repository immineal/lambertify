"""Lambertify — start the web dashboard.

Usage
-----
  python main.py                  # http://0.0.0.0:5000
  python main.py --port 8080
  python main.py --debug
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web.app import app
from src.hardware import query_gpus, query_cpu, recommend_params


def _print_startup_info() -> None:
    gpus = query_gpus()
    cpu  = query_cpu()
    rec  = recommend_params(gpus[0] if gpus else None)

    print("=" * 60)
    print("  Lambertify — AI Music Generator")
    print("=" * 60)
    print(f"  CPU : {cpu.model}")
    print(f"        {cpu.cores_logical} threads | {cpu.ram_total_gb} GB RAM")
    if gpus:
        for g in gpus:
            free_pct = 100 * g.vram_free_mb / max(1, g.vram_total_mb)
            print(f"  GPU{g.index}: {g.name}")
            print(f"        {g.vram_total_mb} MB VRAM | {free_pct:.0f}% free | {g.temperature_c}°C")
    else:
        print("  GPU : none detected — CPU-only mode")
    print()
    print("  Recommended training settings:")
    print(f"    device          = {rec['device']}")
    print(f"    latent_dim      = {rec['latent_dim']}")
    print(f"    vae_batch_size  = {rec['vae_batch_size']}")
    print(f"    lstm_batch_size = {rec['lstm_batch_size']}")
    print(f"    vae_epochs      = {rec['vae_epochs']}")
    for w in rec.get("warnings", []):
        print(f"  ⚠  {w}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Lambertify web server",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--host",  default="0.0.0.0", help="Bind address")
    p.add_argument("--port",  type=int, default=5000, help="Port")
    p.add_argument("--debug", action="store_true", help="Flask debug mode (auto-reload)")
    args = p.parse_args()

    # Suppress Werkzeug's per-request access log — it floods the terminal
    # with every /api/hw poll.  Errors still print because they use WARNING+.
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    _print_startup_info()
    print(f"  Web UI at  http://127.0.0.1:{args.port}")
    print(f"  Press Ctrl-C to stop")
    print()

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=args.debug,
        threaded=True,
    )
