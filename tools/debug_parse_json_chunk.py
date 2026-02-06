import json
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python tools/debug_parse_json_chunk.py <path>")
        return 2
    path = sys.argv[1]
    text = open(path, "r", encoding="utf-8").read()
    s = text.find("{")
    e = text.rfind("}")
    if s < 0 or e <= s:
        print("no json object braces found")
        return 1
    chunk = text[s : e + 1]
    try:
        json.loads(chunk)
        print("OK")
        return 0
    except json.JSONDecodeError as ex:
        print(f"JSONDecodeError: {ex.msg} line={ex.lineno} col={ex.colno} pos={ex.pos}")
        a = max(0, ex.pos - 120)
        b = min(len(chunk), ex.pos + 120)
        snippet = chunk[a:b].replace("\n", "\\n")
        print(snippet)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

