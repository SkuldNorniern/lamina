#!/usr/bin/env python3
import sys
import subprocess
from typing import List

def main() -> int:
    if len(sys.argv) < 3:
        print("usage: with_timeout.py <seconds> <cmd> [args...]", file=sys.stderr)
        return 2
    try:
        timeout_s = float(sys.argv[1])
    except ValueError:
        print("invalid timeout seconds", file=sys.stderr)
        return 2

    cmd: List[str] = sys.argv[2:]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = proc.communicate(timeout=timeout_s)
            sys.stdout.write(out.decode(errors="ignore"))
            sys.stderr.write(err.decode(errors="ignore"))
            return proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                out, err = proc.communicate(timeout=5)
                sys.stdout.write(out.decode(errors="ignore"))
                sys.stderr.write(err.decode(errors="ignore"))
            except Exception:
                pass
            print(f"[TIMEOUT] Command exceeded {timeout_s:.1f}s: {' '.join(cmd)}", file=sys.stderr)
            return 124
    except FileNotFoundError:
        print(f"[ERROR] command not found: {cmd[0]}", file=sys.stderr)
        return 127

if __name__ == "__main__":
    sys.exit(main())





