#!/usr/bin/env python3
"""Apply scipy/numpy compatibility patches to rave/pqmf.py.

Idempotent: safe to run multiple times. Prints what it changed.

Usage:
    python3 patch_pqmf.py /path/to/rave/pqmf.py

Or auto-detect:
    python3 patch_pqmf.py
"""
import sys
import re

def find_pqmf():
    try:
        import rave, os
        return os.path.join(os.path.dirname(rave.__file__), "pqmf.py")
    except ImportError:
        sys.exit("ERROR: acids-rave is not installed.")

path = sys.argv[1] if len(sys.argv) > 1 else find_pqmf()

with open(path) as f:
    src = f.read()

original = src
changes = []

# --- Patch 1: wrap kaiser import for scipy >= 1.14 ---
OLD_IMPORT = "from scipy.signal import kaiser, kaiser_beta"
NEW_IMPORT = (
    "try:\n"
    "    from scipy.signal import kaiser, kaiser_beta\n"
    "except ImportError:\n"
    "    from scipy.signal.windows import kaiser\n"
    "    kaiser_beta = kaiser"
)
if OLD_IMPORT in src and "except ImportError" not in src:
    src = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    changes.append("kaiser import → try/except for scipy >= 1.14")

# --- Patch 2: remove nyq= kwarg from firwin (dropped in scipy 1.14) ---
# Handles both: firwin(N, float(wc)/np.pi, ..., nyq=np.pi)
#           and: firwin(N, wc/np.pi, ..., nyq=np.pi)
src, n = re.subn(r",\s*nyq\s*=\s*np\.pi", "", src)
if n:
    changes.append(f"removed nyq=np.pi kwarg from firwin ({n} occurrence(s))")

# --- Patch 3: replace float(wc) with .item()-safe scalar extraction ---
# Only inside kaiser_filter (don't touch loss_wc or other float() calls)
FLOAT_WC_BODY = re.compile(
    r"(def kaiser_filter\(.*?return h\n)",
    re.DOTALL,
)
def patch_kaiser_body(m):
    body = m.group(1)
    if "wc_scalar" in body:
        return body  # already patched
    body = body.replace(
        "float(wc) / np.pi",
        "wc_scalar / np.pi",
    )
    # Insert wc_scalar assignment after the docstring
    insert = "    wc_scalar = wc.item() if hasattr(wc, 'item') else float(wc)\n"
    # Find the first non-docstring, non-blank line after def
    lines = body.split("\n")
    in_doc = False
    insert_at = None
    for i, line in enumerate(lines[1:], 1):
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            in_doc = not in_doc
            continue
        if in_doc:
            continue
        if stripped and not stripped.startswith("#"):
            insert_at = i
            break
    if insert_at is not None:
        lines.insert(insert_at, insert.rstrip())
        return "\n".join(lines)
    return body

new_src, n = FLOAT_WC_BODY.subn(patch_kaiser_body, src)
if new_src != src:
    src = new_src
    changes.append("float(wc) → wc_scalar (.item()-safe) in kaiser_filter")

if not changes:
    print(f"pqmf.py already patched — nothing to do ({path})")
    sys.exit(0)

with open(path, "w") as f:
    f.write(src)

print(f"Patched {path}:")
for c in changes:
    print(f"  + {c}")
