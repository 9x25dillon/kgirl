#!/usr/bin/env python3
import argparse, json, os, re, sys

KEY_FILENAME_HINTS = [
    "keystore", "wallet", "metamask", "trust", "coinbase",
    "bip39", "seed", "recovery", "backup", "keys", "private"
]
ANDROID_APP_HINTS = [
    "com.metamask", "io.metamask",
    "com.wallet.crypto.trustapp", "com.coinbase.android",
    "com.binance.dev", "com.binance", "com.ledger.live"
]

# A very lightweight heuristic for seed phrases: 12-24 lowercase words a-z
SEED_CANDIDATE = re.compile(r"(?:^|\\b)([a-z]{3,})(?:\\s+[a-z]{3,}){11,23}(?:\\b|$)")

def is_keystore_json(path):
    try:
        if os.path.getsize(path) > 1024 * 1024:  # skip very large jsons
            return False
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            t = f.read(200000)
        # quick sanity: looks like JSON
        if "{" not in t or "}" not in t:
            return False
        # try to parse a small subset safely
        try:
            obj = json.loads(t)
        except Exception:
            # fall back to string heuristics
            pass
        # Heuristics that match Ethereum V3 keystores and common exports
        hits = 0
        for needle in ["\"crypto\"", "\"Cipher\"", "\"cipher\"", "\"kdf\"", "\"scrypt\"", "\"aes-128-ctr\"", "\"pbkdf2\"", "\"address\"", "\"version\""]:
            if needle in t:
                hits += 1
        return hits >= 3
    except Exception:
        return False

def scan(root):
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Flag Android app directories
        for d in dirnames:
            for hint in ANDROID_APP_HINTS:
                if hint.lower() in d.lower():
                    results.append(("android_app_dir", os.path.join(dirpath, d)))
        for fn in filenames:
            lower = fn.lower()
            full = os.path.join(dirpath, fn)
            # Filename hints
            if any(h in lower for h in KEY_FILENAME_HINTS):
                results.append(("name_hit", full))
            # Keystore JSON check
            if lower.endswith(".json"):
                if is_keystore_json(full) or any(h in lower for h in ["keystore", "wallet", "metamask"]):
                    results.append(("json_keystore_like", full))
            # Seed phrase heuristic in small-ish text files
            try:
                if os.path.getsize(full) <= 512 * 1024:
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        txt = f.read()
                    if SEED_CANDIDATE.search(txt):
                        results.append(("seed_phrase_candidate", full))
            except Exception:
                pass
    return results

def main():
    ap = argparse.ArgumentParser(description="Scan a folder for likely crypto wallet artifacts.")
    ap.add_argument("path", help="Folder to scan (e.g., an extracted Android root or Google Drive sync)")
    args = ap.parse_args()
    path = args.path
    if not os.path.isdir(path):
        print(f"Not a directory: {path}", file=sys.stderr)
        sys.exit(2)
    hits = scan(path)
    if not hits:
        print("No obvious artifacts found.")
        return
    # Group by type for readability
    groups = {}
    for kind, p in hits:
        groups.setdefault(kind, []).append(p)
    for kind in ["android_app_dir", "json_keystore_like", "seed_phrase_candidate", "name_hit"]:
        if kind in groups:
            print(f"\n[{kind}] ({len(groups[kind])})")
            for p in sorted(set(groups[kind])):
                print(p)

if __name__ == "__main__":
    main()
