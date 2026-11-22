"""Coverage-based selector: run only the tests that executed the lines a patch changes.

Whenever the coverage map cannot answer for a change, the commit runs every test.
"""

import gzip
import json
import re
import sys
from pathlib import Path

DATA = Path("/data")
MAP_FILE = Path(__file__).resolve().parent / "coverage_map.json.gz"
WORKERS = 4
MISSING, IMPORT = -1, -2
HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
OTHER_CHANGES = ("rename ", "copy ", "new file", "deleted file", "old mode", "new mode", "Binary files", "GIT binary patch")


class Unmapped(Exception):
    """The coverage map cannot say which tests this change affects."""


def split(tests: list[dict], worker_count: int) -> list[list[int]]:
    """Longest test first, each onto the worker with the least to do so far."""
    workers: list[list[int]] = [[] for _ in range(worker_count)]
    loads = [0.0] * worker_count
    for test in sorted(tests, key=lambda test: test["seconds"], reverse=True):
        lightest = loads.index(min(loads))
        workers[lightest].append(test["number"])
        loads[lightest] += test["seconds"]
    return workers


def diff_path(text: str) -> str | None:
    path = text.split("\t")[0].strip()
    if path == "/dev/null":
        return None
    if path.startswith('"'):
        raise Unmapped(f"quoted path {path}")
    return path[2:] if path[:2] in ("a/", "b/") else path


def parse_patch(text: str) -> list[dict]:
    """Each changed file, with its change blocks as (start, deleted, added).

    `start` is the base-file line number where the block begins, `deleted` holds
    (base line number, text) pairs and `added` holds the new lines' text, which sit
    between base lines start - 1 and start + len(deleted).
    """
    files: list[dict] = []
    current: dict | None = None
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith("diff --git "):
            current = {"old": None, "new": None, "blocks": [], "other": False}
            files.append(current)
        elif current is None:
            continue
        elif line.startswith("--- "):
            current["old"] = diff_path(line[4:])
        elif line.startswith("+++ "):
            current["new"] = diff_path(line[4:])
        elif line.startswith(OTHER_CHANGES):
            current["other"] = True
        elif line.startswith("@@"):
            match = HUNK.match(line)
            if not match:
                raise Unmapped(f"unreadable hunk header {line!r}")
            old_number = int(match[1])
            old_left = 1 if match[2] is None else int(match[2])
            new_left = 1 if match[4] is None else int(match[4])
            block = None
            while old_left > 0 or new_left > 0:
                if i >= len(lines):
                    raise Unmapped("hunk ends early")
                body = lines[i]
                i += 1
                tag, content = body[:1], body[1:]
                if tag == "\\":
                    continue
                if tag in (" ", ""):
                    block = None
                    old_number += 1
                    old_left -= 1
                    new_left -= 1
                elif tag in ("-", "+"):
                    if block is None:
                        block = (old_number, [], [])
                        current["blocks"].append(block)
                    if tag == "-":
                        block[1].append((old_number, content))
                        old_number += 1
                        old_left -= 1
                    else:
                        block[2].append(content)
                        new_left -= 1
                else:
                    raise Unmapped(f"unreadable hunk line {body!r}")
    return files


def is_blank_or_comment(text: str) -> bool:
    stripped = text.strip()
    return not stripped or stripped.startswith("#")


def tests_at(coverage: dict, path: str, number: int) -> list[int]:
    value = coverage["files"][path].get(str(number))
    if value is None:
        raise Unmapped(f"{path}:{number} is not part of a statement")
    if value == MISSING:
        raise Unmapped(f"{path}:{number} was never run by a test")
    if value == IMPORT:
        raise Unmapped(f"{path}:{number} runs at import time or outside a test")
    return coverage["sets"][value]


def tests_for_patch(text: str, coverage: dict) -> set[int]:
    """Indexes into coverage["tests"] of every test that ran a changed base line."""
    files = parse_patch(text)
    if not files:
        raise Unmapped("no files in patch")

    selected: set[int] = set()
    for changed in files:
        path = changed["new"]
        if changed["other"] or changed["old"] is None or changed["old"] != path:
            raise Unmapped(f"{changed['old']} -> {path} is added, deleted, renamed or a mode change")
        fixture = coverage.get("fixtures", {}).get(path)
        if fixture is not None:
            if fixture < 0:
                raise Unmapped(f"{path} is read outside a test")
            selected.update(coverage["sets"][fixture])
            continue
        if path not in coverage["files"]:
            raise Unmapped(f"{path} is not in the coverage map")
        lines = coverage["files"][path]

        for start, deleted, added in changed["blocks"]:
            deleted_code = False
            for number, content in deleted:
                if str(number) not in lines and is_blank_or_comment(content):
                    continue
                selected.update(tests_at(coverage, path, number))
                deleted_code = True

            if not added or deleted_code:
                continue
            before, after = start - 1, start + len(deleted)
            # A comment or blank line outside any statement cannot change what runs.
            outside_statement = str(before) not in lines or str(after) not in lines
            if outside_statement and all(is_blank_or_comment(line) for line in added):
                continue
            # New code runs where its neighbours run.
            selected.update(tests_at(coverage, path, before))
            selected.update(tests_at(coverage, path, after))
    return selected


def main() -> None:
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    every_test = split(tests, WORKERS)
    try:
        coverage = json.loads(gzip.decompress(MAP_FILE.read_bytes()))
        by_id = {test["id"]: test for test in tests}
        known = [by_id.get(test_id) for test_id in coverage["tests"]]
    except Exception as error:
        print(f"no coverage map ({error}); every commit runs every test", file=sys.stderr)
        coverage = None

    for patch in sorted((DATA / "commits").glob("*.patch")):
        workers = every_test
        if coverage is not None:
            try:
                chosen = tests_for_patch(patch.read_text(encoding="utf-8", errors="replace"), coverage)
                picked = sorted((known[i] for i in chosen if known[i] is not None), key=lambda t: t["number"])
                workers = [worker for worker in split(picked, WORKERS) if worker]
                print(f"{patch.stem}: {len(picked)} tests", file=sys.stderr)
            except Exception as reason:
                print(f"{patch.stem}: every test ({reason})", file=sys.stderr)
        print(json.dumps({"commit": patch.stem, "workers": workers}))


if __name__ == "__main__":
    main()
