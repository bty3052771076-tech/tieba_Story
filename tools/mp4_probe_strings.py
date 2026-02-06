import pathlib
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python tools/mp4_probe_strings.py <path>")
        return 2
    p = pathlib.Path(sys.argv[1])
    b = p.read_bytes()
    print("soun", b.find(b"soun"))
    print("mp4a", b.find(b"mp4a"))
    print("Opus", b.find(b"Opus"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

