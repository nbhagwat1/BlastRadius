"""Build coverage_map.json.gz: which pytest tests executed each line of sqlglot's source.

Run once from the repository root, inside the image the scorer uses:

    MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD:/work" -w /work \
        python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 \
        sh -c "pip install -q -r requirements.lock pytest-cov && python build_coverage_map.py"

Map format: {"tests": [test ids], "sets": [[indexes into tests]], "files": {path: {line: value}},
"fixtures": {path: value}}. A line's value is an index into "sets", MISSING (no test ran it) or
IMPORT (it runs at import time or outside any test). Lines that belong to no statement (blank,
comment, docstring) are absent. "fixtures" lists the data files under tests/ that the tests opened.
"""

import ast
import concurrent.futures
import gzip
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE / "sqlglot"
TESTS_DIR = str(REPO / "tests") + os.sep
OUT = HERE / "coverage_map.json.gz"
SOURCE_OUT = HERE / "sqlglot_source.json.gz"
DATA_FILE = "/tmp/blast-radius.coverage"
READS_FILE = "/tmp/blast-radius-reads.json"
MISSING, IMPORT = -1, -2
PHASES = ("setup", "run", "teardown")
# The package __init__.py files share no dialect, so they must not form a group of their own.
GROUP_RE = re.compile(r"sqlglot/(dialects|parsers|generators|typing)/(?!__init__\.py)(\w+)\.py$")
WORD = re.compile(r"[A-Z0-9_]+")

# Data files under tests/ that were opened -> contexts that opened them, in pytest-cov's
# "<test id>|<phase>" form, or "" when no test was running.
_current_context = ""
_reads: dict[str, set[str]] = {}


def _record_open(event, args):
    if event != "open" or not isinstance(args[0], (str, os.PathLike)):
        return
    path = os.path.abspath(os.fspath(args[0]))
    if path.startswith(TESTS_DIR) and not path.endswith((".py", ".pyc")):
        relative = Path(path).relative_to(REPO).as_posix()
        _reads.setdefault(relative, set()).add(_current_context)


def _in_phase(item, phase):
    global _current_context
    _current_context = f"{item.nodeid}|{phase}"
    try:
        return (yield)
    finally:
        _current_context = ""


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    return (yield from _in_phase(item, "setup"))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    return (yield from _in_phase(item, "run"))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    return (yield from _in_phase(item, "teardown"))


def pytest_unconfigure(config):
    Path(READS_FILE).write_text(json.dumps({path: sorted(c) for path, c in _reads.items()}))


class InProcessPool(concurrent.futures.Executor):
    """Runs each job at once in this process, so coverage credits it to the running test.

    Jobs and results still go through pickle, as with real processes, because pickling
    sqlglot expressions runs sqlglot code (serde.py) that those tests depend on.
    """

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kwargs):
        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, fn, /, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            future.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
        except Exception as error:
            future.set_exception(error)
        return future


def pytest_configure(config):
    # TestOptimizer and TestExecutor send their sqlglot work to spawned processes, which
    # coverage would record without the test's context. Patch before test modules import it.
    concurrent.futures.ProcessPoolExecutor = InProcessPool
    sys.addaudithook(_record_open)


def pytest_collection_finish(session):
    # Import every sqlglot module before the first test starts, so import-time lines are
    # recorded with no test context instead of under whichever test happened to import them.
    import importlib
    import pkgutil

    import sqlglot

    for info in pkgutil.walk_packages(sqlglot.__path__, "sqlglot.", onerror=lambda name: None):
        if info.name != "sqlglot.__main__":
            try:
                importlib.import_module(info.name)
            except Exception:
                pass


def run_tests() -> None:
    env = dict(
        os.environ,
        SKIP_INTEGRATION="0",
        COVERAGE_FILE=DATA_FILE,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join([str(REPO), str(HERE)]),
    )
    command = [
        sys.executable, "-m", "pytest", "tests", "-q",
        "-p", "no:cacheprovider", "-p", "build_coverage_map",
        "--cov=sqlglot", "--cov-context=test", "--cov-report=",
    ]
    code = subprocess.run(command, cwd=REPO, env=env).returncode
    print(f"pytest exit code {code}", file=sys.stderr)


def statement_groups(path: Path, source: str) -> dict[int, list[int]]:
    """First line of each statement -> every physical line the statement spans."""
    from coverage.parser import PythonParser

    parser = PythonParser(text=source, filename=str(path))
    parser.parse_source()
    groups: dict[int, list[int]] = {}
    for line in range(1, source.count("\n") + 2):
        first = parser.first_line(line)
        if first in parser.statements:
            groups.setdefault(first, []).append(line)
    return groups


def function_body_lines(source: str) -> set[int]:
    lines = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for statement in node.body:
                lines.update(range(statement.lineno, statement.end_lineno + 1))
    return lines


def build_map() -> None:
    from coverage import CoverageData

    data = CoverageData(DATA_FILE)
    data.read()

    class_tests: dict[str, set[str]] = {}
    for context in data.measured_contexts():
        test_id, _, phase = context.rpartition("|")
        if phase in PHASES:
            class_tests.setdefault(test_id.rpartition("::")[0], set()).add(test_id)

    def tests_for(context: str) -> set[str] | None:
        test_id, _, phase = context.rpartition("|")
        if phase not in PHASES:
            return None
        # setUpClass / tearDownClass run inside the first / last test's setup / teardown,
        # but a change to what they run affects every test in the class.
        if phase in ("setup", "teardown"):
            return class_tests[test_id.rpartition("::")[0]]
        return {test_id}

    test_index: dict[str, int] = {}
    set_index: dict[tuple[int, ...], int] = {}
    files: dict[str, dict[str, int]] = {}
    counts = {"tested": 0, "missing": 0, "import": 0}

    def set_number(tests: set[str]) -> int:
        key = tuple(sorted(test_index.setdefault(t, len(test_index)) for t in sorted(tests)))
        return set_index.setdefault(key, len(set_index))

    for measured in sorted(data.measured_files()):
        path = Path(measured).resolve()
        try:
            relative = path.relative_to(REPO).as_posix()
        except ValueError:
            continue
        if not relative.startswith("sqlglot/") or not relative.endswith(".py"):
            continue

        source = path.read_text(encoding="utf-8")
        in_function = function_body_lines(source)
        by_line = data.contexts_by_lineno(measured)
        entries: dict[str, int] = {}

        for first, lines in statement_groups(path, source).items():
            contexts = set()
            for line in lines:
                contexts.update(by_line.get(line, ()))
            tests: set[str] = set()
            outside_tests = first not in in_function
            for context in contexts:
                found = tests_for(context)
                if found is None:
                    outside_tests = True
                else:
                    tests |= found

            if outside_tests:
                value = IMPORT
                counts["import"] += 1
            elif not tests:
                value = MISSING
                counts["missing"] += 1
            else:
                value = set_number(tests)
                counts["tested"] += 1
            for line in lines:
                entries[str(line)] = value
        files[relative] = entries

    # Data files under tests/: a file read outside any test is IMPORT, like import-time code.
    fixtures: dict[str, int] = {}
    for relative, contexts in sorted(json.loads(Path(READS_FILE).read_text()).items()):
        found = [tests_for(context) for context in contexts]
        if any(tests is None for tests in found):
            fixtures[relative] = IMPORT
        else:
            fixtures[relative] = set_number(set().union(*found))

    # Reverse lookups for building aggregate test sets.
    test_list = sorted(test_index, key=test_index.__getitem__)
    sets_by_id = {v: k for k, v in set_index.items()}

    def tests_in_file(entries: dict[str, int]) -> set[str]:
        result: set[str] = set()
        for val in entries.values():
            if val >= 0:
                for idx in sets_by_id[val]:
                    result.add(test_list[idx])
        return result

    # Module groups: dialect/parser/generator/typing files sharing a dialect name.
    groups: dict[str, list[str]] = {}
    for path in files:
        m = GROUP_RE.match(path)
        if m:
            groups.setdefault(m.group(2), []).append(path)

    module_groups: dict[str, int] = {}
    for dialect, paths in groups.items():
        all_tests: set[str] = set()
        for path in paths:
            all_tests |= tests_in_file(files[path])
        # Include tests from the matching test file (tests/dialects/test_<dialect>.py).
        test_prefix = f"tests/dialects/test_{dialect}.py::"
        all_tests |= {t for t in test_index if t.startswith(test_prefix)}
        if all_tests:
            sn = set_number(all_tests)
            for path in paths:
                module_groups[path] = sn

    # File-level test union: for each file, the set of all tests that run any line in it.
    file_tests: dict[str, int] = {}
    for path, entries in files.items():
        ft = tests_in_file(entries)
        if ft:
            file_tests[path] = set_number(ft)

    # Functions using importlib.import_module() may be under-recorded: the coverage build
    # eagerly imports all modules, so lazy loaders like _try_load are only called by the
    # few tests that explicitly test lazy loading.  Widen their tested lines to the module group.
    widen_to_group: dict[str, list[list[int]]] = {}
    for path in files:
        if path not in module_groups:
            continue
        source = (REPO / path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in ast.walk(node):
                if (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "import_module"):
                    widen_to_group.setdefault(path, []).append([node.body[0].lineno, node.end_lineno])
                    break

    payload = {
        "tests": sorted(test_index, key=test_index.__getitem__),
        "sets": [list(key) for key in sorted(set_index, key=set_index.__getitem__)],
        "files": files,
        "fixtures": fixtures,
        "module_groups": module_groups,
        "file_tests": file_tests,
        "widen_to_group": widen_to_group,
    }
    OUT.write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), mtime=0))

    expected = {t["id"] for t in json.loads((HERE / "sample" / "tests.json").read_text())["tests"]}
    recorded = set().union(*class_tests.values())
    print(f"statements: {counts}", file=sys.stderr)
    print(f"files: {len(files)}, test sets: {len(set_index)}, size: {OUT.stat().st_size} bytes", file=sys.stderr)
    print(f"tests.json ids recorded: {len(expected & recorded)}/{len(expected)}", file=sys.stderr)
    for missing in sorted(expected - recorded):
        print(f"  not recorded: {missing}", file=sys.stderr)
    for relative, value in fixtures.items():
        readers = "read outside a test" if value == IMPORT else f"{len(payload['sets'][value])} tests"
        print(f"fixture {relative}: {readers}", file=sys.stderr)


def dialect_families() -> dict[str, set[str]]:
    """Dialect name -> itself and every dialect whose class inherits from it."""
    parent: dict[str, str | None] = {}
    for path in (REPO / "sqlglot" / "dialects").glob("*.py"):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef) and node.name.lower() == path.stem:
                bases = [b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases]
                parent[path.stem] = next((b.lower() for b in bases if b.lower() != "dialect"), None)
    families = {stem: {stem} for stem in parent}
    for stem in parent:
        ancestor = parent[stem]
        while ancestor:
            families.setdefault(ancestor, {ancestor}).add(stem)
            ancestor = parent.get(ancestor)
    return families


def add_selection_tables() -> None:
    """Add the tables the selector narrows import-time changes with, built from the map and the pinned tests.

    "families": grouped dialect file -> tests of its dialect group, of the groups of dialects that
    inherit from it, and of tests or fixtures naming any of those dialects.
    "word_tests": upper-cased word -> tests whose own function source, or a fixture they read, contains it.
    """
    payload = json.loads(gzip.decompress(OUT.read_bytes()))
    sets = payload["sets"]
    set_index = {tuple(s): i for i, s in enumerate(sets)}

    def set_number(tests: set[int]) -> int:
        key = tuple(sorted(tests))
        if key not in set_index:
            set_index[key] = len(sets)
            sets.append(list(key))
        return set_index[key]

    payload["module_groups"] = {p: g for p, g in payload["module_groups"].items() if GROUP_RE.match(p)}
    payload["widen_to_group"] = {p: r for p, r in payload["widen_to_group"].items() if p in payload["module_groups"]}

    index = {test_id: i for i, test_id in enumerate(payload["tests"])}
    texts: dict[int, str] = {}
    for file in sorted({test_id.split("::")[0] for test_id in payload["tests"]}):
        source = (REPO / file).read_text(encoding="utf-8")
        tree = ast.parse(source)
        scopes = [("", tree.body)] + [(node.name + "::", node.body) for node in tree.body if isinstance(node, ast.ClassDef)]
        for prefix, body in scopes:
            for node in body:
                test_id = f"{file}::{prefix}{node.name}" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
                if test_id in index:
                    texts[index[test_id]] = (ast.get_source_segment(source, node) or "").upper()
    fixtures = [(set(WORD.findall((REPO / path).read_text(encoding="utf-8", errors="replace").upper())), set(sets[value]))
                for path, value in payload["fixtures"].items() if value >= 0]

    words: dict[str, set[int]] = {}
    for test, text in texts.items():
        for word in set(WORD.findall(text)):
            words.setdefault(word, set()).add(test)
    for fixture_words, readers in fixtures:
        for word in fixture_words:
            words.setdefault(word, set()).update(readers)
    payload["word_tests"] = {word: set_number(tests) for word, tests in sorted(words.items())}

    groups_by_name: dict[str, set[int]] = {}
    for path, group in payload["module_groups"].items():
        groups_by_name.setdefault(GROUP_RE.match(path).group(2), set()).update(sets[group])
    families = dialect_families()
    payload["families"] = {}
    for path in payload["module_groups"]:
        name = GROUP_RE.match(path).group(2)
        if name == "dialect":
            continue
        found: set[int] = set()
        for member in families.get(name, {name}):
            found |= groups_by_name.get(member, set())
            quoted = re.compile(r"[\"']" + re.escape(member.upper()) + r"[\"']")
            found |= {test for test, text in texts.items() if quoted.search(text)}
            for fixture_words, readers in fixtures:
                if member.upper() in fixture_words:
                    found |= readers
        payload["families"][path] = set_number(found)

    payload["arg_flips"] = arg_flips(payload, set_number)

    OUT.write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), mtime=0))
    print(f"arg_types spans: {sum(map(len, payload['arg_flips'].values()))}, "
          f"words: {len(payload['word_tests'])}, families: {len(payload['families'])}, "
          f"sets: {len(sets)}, size: {OUT.stat().st_size} bytes", file=sys.stderr)


def arg_flips(payload: dict, set_number) -> dict[str, list[list[int]]]:
    """expressions file -> [first line, last line, tests] of each non-Func class's arg_types dict.

    Flipping one of its True/False values only changes the class's required_args, which only the
    parser's validate_expression reads. So it reaches the tests running parser code that names the
    class or a subclass; a Func is also built and validated by name, so it gets no entry.
    """
    import selector

    sources = {p.relative_to(REPO).as_posix(): p.read_text(encoding="utf-8") for p in sorted((REPO / "sqlglot").rglob("*.py"))}
    bases: dict[str, set[str]] = {}
    spans = []
    for path, source in sources.items():
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef):
                continue
            bases.setdefault(node.name, set()).update(b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases)
            if path.startswith("sqlglot/expressions/"):
                for statement in node.body:
                    targets = statement.targets if isinstance(statement, ast.Assign) else [getattr(statement, "target", None)]
                    if isinstance(statement, (ast.Assign, ast.AnnAssign)) and any(getattr(t, "id", "") == "arg_types" for t in targets):
                        spans.append((path, statement.lineno, statement.end_lineno, node.name))

    def ancestors(name: str) -> set[str]:
        found, todo = set(), [name]
        while todo:
            for base in bases.get(todo.pop(), ()):
                if base not in found:
                    found.add(base)
                    todo.append(base)
        return found

    parser_files = re.compile(r"sqlglot/(parser\.py|parsers/\w+\.py|dialects/\w+\.py)$")
    flips: dict[str, list[list[int]]] = {}
    for path, first, last, name in spans:
        family = {c for c in bases if c == name or name in ancestors(c)}
        if any(c == "Func" or "Func" in ancestors(c) for c in family):
            continue
        mentions = re.compile(r"\b(" + "|".join(sorted(map(re.escape, family))) + r")\b")
        found: set[int] = set()
        try:
            for parser_path, source in sources.items():
                if not parser_files.match(parser_path):
                    continue
                lines = payload["files"].get(parser_path, {})
                for number, line in enumerate(source.split("\n"), 1):
                    if mentions.search(line):
                        value = lines.get(str(number))
                        if value == IMPORT:
                            found |= set(selector.import_tests(payload, parser_path, number, [line]))
                        elif value is not None and value >= 0:
                            found |= set(payload["sets"][value])
        except selector.Unmapped:
            continue
        flips.setdefault(path, []).append([first, last, set_number(found)])
    return flips


def write_sources() -> None:
    """The pinned commit's sqlglot/*.py, so the selector can compile a patched file without importing it."""
    sources = {
        # read_text's universal newlines turn a CRLF checkout back into the commit's LF lines.
        path.relative_to(REPO).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted((REPO / "sqlglot").rglob("*.py"))
    }
    SOURCE_OUT.write_bytes(gzip.compress(json.dumps(sources, separators=(",", ":")).encode(), mtime=0))
    print(f"sources: {len(sources)} files, size: {SOURCE_OUT.stat().st_size} bytes", file=sys.stderr)


if __name__ == "__main__":
    if sys.argv[1:] != ["--tables"]:
        run_tests()
        build_map()
    add_selection_tables()
    write_sources()
