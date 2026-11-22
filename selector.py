"""Coverage-based selector: run only the tests that executed the lines a patch changes.

Whenever the coverage map cannot answer for a change, the commit runs every test.
"""

import ast
import gzip
import json
import re
import sys
import warnings
from pathlib import Path

DATA = Path("/data")
MAP_FILE = Path(__file__).resolve().parent / "coverage_map.json.gz"
SOURCE_FILE = Path(__file__).resolve().parent / "sqlglot_source.json.gz"
MAX_WORKERS = 8
WORKER_OVERHEAD_MS = 2000
MISSING, IMPORT = -1, -2
HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
QUOTED = re.compile(r'"([^"\\]{2,})"')
BOOL = re.compile(r"\b(True|False)\b")
TOKEN = re.compile(r"\bis\s+not\b|\w+|==|!=|[^\w\s]")
EQUALITY = {"is": "==", "is not": "!=", "==": "==", "!=": "!="}
WORD = re.compile(r"[A-Z0-9_]+")
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


def add_to_lightest(workers: list[list[int]], extra: list[dict], seconds: dict[int, float]) -> None:
    """Longest first, each extra test onto the existing worker with the least to do."""
    loads = [sum(seconds[number] for number in worker) for worker in workers]
    present = {number for worker in workers for number in worker}
    for test in sorted(extra, key=lambda test: test["seconds"], reverse=True):
        if test["number"] in present:
            continue
        lightest = loads.index(min(loads))
        workers[lightest].append(test["number"])
        loads[lightest] += test["seconds"]
        present.add(test["number"])


def cheapest_split(tests: list[dict], extra: list[dict], seconds: dict[int, float]) -> list[list[int]]:
    """The worker count, 1 to MAX_WORKERS, whose busiest worker plus per-worker overhead costs least."""
    best, best_cost = [], None
    for count in range(1, MAX_WORKERS + 1):
        workers = [worker for worker in split(tests, count) if worker]
        if workers and extra:
            add_to_lightest(workers, extra, seconds)
        busiest = max((sum(int(seconds[n] * 1000 + 0.5) for n in worker) for worker in workers), default=0)
        cost = busiest + WORKER_OVERHEAD_MS * len(workers)
        if best_cost is None or cost < best_cost:
            best, best_cost = workers, cost
    return best


def changes_source(text: str, coverage: dict) -> bool:
    return any(changed["new"] in coverage["files"] for changed in parse_patch(text))


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


_sources: dict[str, str] | None = None


def pinned_source(path: str) -> str:
    global _sources
    if _sources is None:
        try:
            _sources = json.loads(gzip.decompress(SOURCE_FILE.read_bytes()))
        except (OSError, ValueError) as error:
            raise Unmapped(f"no pinned source to compile against ({error})")
    if path not in _sources:
        raise Unmapped(f"{path} has no pinned source to compile against")
    return _sources[path]


def patched_source(path: str, blocks: list) -> str:
    """The file after the patch, built from the pinned source; any disagreement with it is Unmapped."""
    lines = pinned_source(path).split("\n")
    for start, deleted, added in reversed(blocks):
        if start < 1 or start - 1 + len(deleted) > len(lines):
            raise Unmapped(f"{path}:{start} is past the end of the pinned source")
        for offset, (number, content) in enumerate(deleted):
            if number != start + offset or lines[number - 1] != content.rstrip("\r"):
                raise Unmapped(f"{path}:{number} does not match the pinned source")
        lines[start - 1 : start - 1 + len(deleted)] = [line.rstrip("\r") for line in added]
    return "\n".join(lines)


def check_compiles(path: str, blocks: list) -> None:
    """A file that no longer compiles breaks every test that imports it, however indirectly."""
    source = patched_source(path, blocks)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile(source, path, "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as error:
        raise Unmapped(f"{path} does not compile after the patch: {error}")


def is_blank_or_comment(text: str) -> bool:
    stripped = text.strip()
    return not stripped or stripped.startswith("#")


def quoted_keys(texts: list[str]) -> list[str]:
    """Quoted strings on the changed lines that look like SQL words, e.g. the "ANTI" of a keyword table entry."""
    return [key for text in texts for key in QUOTED.findall(text)
            if re.search(r"[A-Za-z]", key) and not key.startswith(("%", " "))]


def key_tests(coverage: dict, keys: list[str]) -> set[int]:
    """Tests whose source, or a fixture they read, contains every word of one of the keys."""
    found: set[int] = set()
    for key in {key.upper() for key in keys}:
        tests: set[int] | None = None
        for word in WORD.findall(key):
            value = coverage["word_tests"].get(word)
            matching = set() if value is None else set(coverage["sets"][value])
            tests = matching if tests is None else tests & matching
        found |= tests or set()
    return found


def import_tests(coverage: dict, path: str, number: int, texts: list[str]) -> list[int] | set[int]:
    """An import-time line affects whoever uses what it defines, which coverage cannot see.

    A dialect file's line reaches its dialect, the dialects inheriting from it and the tests naming
    them; a core table entry keyed by an upper-case SQL word reaches the tests using that word.
    Anything else might affect every test.
    """
    family = coverage["families"].get(path)
    if family is not None:
        return coverage["sets"][family]
    group = coverage["module_groups"].get(path)
    if group is not None:
        return coverage["sets"][group]
    keys = quoted_keys(texts)
    if any(key.upper() == key for key in keys):
        return key_tests(coverage, keys)
    raise Unmapped(f"{path}:{number} runs at import time or outside a test")


def arg_flip_tests(coverage: dict, path: str, deleted: list, added: list) -> list[int] | None:
    """Tests a one-line True/False flip inside a non-Func class's arg_types can affect; None for any other block."""
    if len(deleted) != 1 or len(added) != 1:
        return None
    (number, old), new = deleted[0], added[0]
    if old == new or BOOL.sub("B", old) != BOOL.sub("B", new):
        return None
    for first, last, tests in coverage.get("arg_flips", {}).get(path, ()):
        if first <= number <= last:
            return coverage["sets"][tests]
    return None


def attr_read_tests(coverage: dict, path: str, deleted: list, added: list) -> list[int] | None:
    """Tests an in-place edit of one import-time assignment in a dialect file can affect; None for any other block."""
    if not deleted or len(deleted) != len(added):
        return None
    lines = coverage["files"][path]
    for first, last, tests in coverage.get("attr_reads", {}).get(path, ()):
        if all(first <= number <= last and lines.get(str(number)) == IMPORT for number, _ in deleted):
            for (number, old), new in zip(deleted, added):
                if not same_layout(old, new):
                    return None
                if number == first and old.split("=")[0] != new.split("=")[0]:
                    return None
            return coverage["sets"][tests]
    return None


def same_comparison(old: str, new: str) -> bool:
    """`is None` <-> `== None` and `is X.MEMBER` -> `== X.MEMBER` (or the negated forms): the same test either way."""
    before = [re.sub(r"\s+", " ", token) for token in TOKEN.findall(old)]
    after = [re.sub(r"\s+", " ", token) for token in TOKEN.findall(new)]
    if before == after or len(before) != len(after):
        return False
    for i, (a, b) in enumerate(zip(before, after)):
        if a == b:
            continue
        if a not in EQUALITY or b not in EQUALITY or EQUALITY[a] != EQUALITY[b]:
            return False
        operand = before[i + 1 : i + 2]
        j = i + 1
        if operand == ["None"]:
            j += 1
        else:
            # A dotted enum member: Name(.name)*.UPPER_CASE. Only identity -> equality: a str-valued
            # enum compared by == can match a raw string, which `is` would then reject.
            if a not in ("is", "is not") or j >= len(before) or not before[j][:1].isupper():
                return False
            j += 1
            while j + 1 < len(before) and before[j] == ".":
                j += 2
            if before[j - 2 : j - 1] != ["."] or not re.fullmatch(r"[A-Z][A-Z0-9_]*", before[j - 1]):
                return False
        if j < len(before) and before[j] in (".", "(", "["):
            return False
    return True


def scope_effects(text: str) -> set | None:
    """What a statement does to its whole function, wherever it sits: the names it binds, its
    global/nonlocal declarations, and whether it makes the function a generator or coroutine.
    None when the line is not a statement on its own."""
    source = text.strip()
    if source.endswith(":"):
        source += " pass"
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    effects = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            effects.add(("bind", node.id))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            effects.update(("declare", name) for name in node.names)
        elif isinstance(node, ast.alias):
            effects.add(("bind", (node.asname or node.name).split(".")[0]))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            effects.add(("bind", node.name))
        elif isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await)):
            effects.add(("kind", type(node).__name__))
    return effects


def same_layout(old: str, new: str) -> bool:
    """The replacement cannot move other lines into or out of a block or statement."""
    indent = lambda text: text[: len(text) - len(text.lstrip())]
    depth = lambda text: sum(text.count(c) for c in "([{") - sum(text.count(c) for c in ")]}")
    ending = lambda text: text.rstrip()[-1:] in (":", "\\")
    return (indent(old) == indent(new) and depth(old) == depth(new) and ending(old) == ending(new)
            and not any(q in old + new for q in ('"""', "'''")))


def same_shape(old: str, new: str) -> bool:
    """Same layout, and the replacement cannot change its function's scope either."""
    if not same_layout(old, new):
        return False
    effects = scope_effects(old)
    return effects is not None and effects == scope_effects(new)


def tests_at(coverage: dict, path: str, number: int, texts: list[str]) -> list[int] | set[int]:
    value = coverage["files"][path].get(str(number))
    if value is None:
        raise Unmapped(f"{path}:{number} is not part of a statement")
    if value == IMPORT:
        return import_tests(coverage, path, number, texts)
    if value == MISSING:
        group = coverage.get("module_groups", {}).get(path)
        if group is not None:
            return coverage["sets"][group]
        fallback = coverage.get("file_tests", {}).get(path)
        if fallback is not None:
            return coverage["sets"][fallback]
        raise Unmapped(f"{path}:{number} has no test coverage and no module group")
    for start, end in coverage.get("widen_to_group", {}).get(path, ()):
        if start <= number <= end:
            group = coverage.get("module_groups", {}).get(path)
            if group is not None:
                return coverage["sets"][group]
            break
    return coverage["sets"][value]


def tests_for_patch(text: str, coverage: dict) -> tuple[set[int], bool]:
    """Indexes into coverage["tests"] of every test that ran a changed base line, and whether any changed line runs at import time."""
    files = parse_patch(text)
    if not files:
        raise Unmapped("no files in patch")

    selected: set[int] = set()
    import_time = False
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
        check_compiles(path, changed["blocks"])
        lines = coverage["files"][path]

        for start, deleted, added in changed["blocks"]:
            # A block that only touches comments and blank lines changes no behaviour.
            if all(is_blank_or_comment(content) for _, content in deleted) and all(map(is_blank_or_comment, added)):
                continue
            if len(deleted) == len(added) and all(
                content == new or same_comparison(content, new) for (_, content), new in zip(deleted, added)
            ):
                continue
            # No test ever reaches a MISSING line, so rewriting one in place changes no test's run.
            if len(deleted) == len(added) and all(
                lines.get(str(number)) == MISSING and same_shape(content, new)
                for (number, content), new in zip(deleted, added)
            ):
                continue
            flipped = arg_flip_tests(coverage, path, deleted, added)
            if flipped is None:
                flipped = attr_read_tests(coverage, path, deleted, added)
            if flipped is not None:
                selected.update(flipped)
                import_time = True
                continue
            texts = [content for _, content in deleted] + added
            deleted_code = False
            for number, content in deleted:
                if str(number) not in lines and is_blank_or_comment(content):
                    continue
                import_time |= lines.get(str(number)) == IMPORT
                selected.update(tests_at(coverage, path, number, texts))
                deleted_code = True

            if not added or deleted_code:
                continue
            before, after = start - 1, start + len(deleted)
            # A comment or blank line outside any statement cannot change what runs.
            outside_statement = str(before) not in lines or str(after) not in lines
            only_comments = all(is_blank_or_comment(line) for line in added)
            if outside_statement and only_comments:
                continue
            # Nor can one between two statements that only run at import time.
            if only_comments and lines.get(str(before)) == IMPORT and lines.get(str(after)) == IMPORT:
                continue
            # New code runs where its neighbours run.
            import_time |= IMPORT in (lines.get(str(before)), lines.get(str(after)))
            selected.update(tests_at(coverage, path, before, texts))
            selected.update(tests_at(coverage, path, after, texts))
    return selected, import_time


# Uncovered tests that only see what runs at import time: the module-level code a fresh
# `import sqlglot` runs, sqlglotc/setup.py, and the typing modules' module-level metadata.
IMPORT_ONLY_TESTS = {
    "tests/dialects/test_dialect.py::TestDialect::test_lazy_load",
    "tests/test_sqlglotc.py::TestSQLGlotC::test_build_removes_obsolete_extensions",
    "tests/test_sqlglotc.py::TestSQLGlotC::test_dry_run_preserves_obsolete_extensions",
    "tests/test_sqlglotc.py::TestSQLGlotC::test_editable_install_removes_obsolete_extensions",
    "tests/test_typing.py::TestTyping::test_no_redundant_expression_metadata",
}


def unmapped_for(unmapped: list[dict], import_time: bool) -> list[dict]:
    return unmapped if import_time else [test for test in unmapped if test["id"] not in IMPORT_ONLY_TESTS]


def main() -> None:
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    seconds = {test["number"]: test["seconds"] for test in tests}
    every_test = cheapest_split(tests, [], seconds)
    try:
        coverage = json.loads(gzip.decompress(MAP_FILE.read_bytes()))
        by_id = {test["id"]: test for test in tests}
        known = [by_id.get(test_id) for test_id in coverage["tests"]]
        # Tests that recorded no coverage: the map can never pick them, so any source change might break them.
        mapped = set(coverage["tests"])
        unmapped = [test for test in tests if test["id"] not in mapped]
        print(f"{len(unmapped)} tests have no coverage", file=sys.stderr)
    except Exception as error:
        print(f"no coverage map ({error}); every commit runs every test", file=sys.stderr)
        coverage = None

    for patch in sorted((DATA / "commits").glob("*.patch")):
        workers = every_test
        if coverage is not None:
            try:
                text = patch.read_text(encoding="utf-8", errors="replace")
                chosen, import_time = tests_for_patch(text, coverage)
                picked = sorted((known[i] for i in chosen if known[i] is not None), key=lambda t: t["number"])
                extra = unmapped_for(unmapped, import_time) if changes_source(text, coverage) else []
                workers = cheapest_split(picked, extra, seconds)
                print(f"{patch.stem}: {sum(map(len, workers))} tests", file=sys.stderr)
            except Exception as reason:
                print(f"{patch.stem}: every test ({reason})", file=sys.stderr)
        print(json.dumps({"commit": patch.stem, "workers": workers}))


if __name__ == "__main__":
    main()
