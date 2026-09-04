"""Generate a reproducible seeded-bug task set.

Each task is a small self-contained repository holding one deliberate defect and a
test suite that catches it. These are not real-world issues; they are a controlled
suite for comparing control-loop changes cheaply and offline. Report them as a
seeded-bug benchmark, never as SWE-bench.

    python benchmarks/build_local_tasks.py
    coding-agent evaluate --tasks benchmarks/tasks.local.json --output runs/seeded-v1
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "slugify-trailing-separators",
        "issue": (
            "slugify() leaves a leading and trailing hyphen when the input starts or "
            "ends with punctuation or whitespace. slugify('  Hello, World!  ') returns "
            "'-hello-world-' but should return 'hello-world'."
        ),
        "files": {
            "slugify.py": '''import re


def slugify(text):
    """Convert arbitrary text into a URL-safe slug."""

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text
''',
            "test_slugify.py": '''from slugify import slugify


def test_simple_phrase():
    assert slugify("Hello World") == "hello-world"


def test_collapses_runs_of_separators():
    assert slugify("a---b") == "a-b"


def test_strips_leading_and_trailing_separators():
    assert slugify("  Hello, World!  ") == "hello-world"
''',
        },
    },
    {
        "task_id": "roman-subtractive-notation",
        "issue": (
            "to_roman() never emits subtractive pairs. to_roman(4) returns 'IIII' "
            "instead of 'IV', and to_roman(2024) returns 'MMXXIIII' instead of "
            "'MMXXIV'. The 4, 9, 40, 90, 400 and 900 cases are all wrong."
        ),
        "files": {
            "roman.py": '''VALUES = [
    (1000, "M"),
    (500, "D"),
    (100, "C"),
    (50, "L"),
    (10, "X"),
    (5, "V"),
    (1, "I"),
]


def to_roman(number):
    """Convert an integer in 1..3999 into a Roman numeral."""

    if not 1 <= number <= 3999:
        raise ValueError("number must be between 1 and 3999")
    digits = []
    for value, symbol in VALUES:
        while number >= value:
            digits.append(symbol)
            number -= value
    return "".join(digits)
''',
            "test_roman.py": '''import pytest

from roman import to_roman


def test_additive_values():
    assert to_roman(3) == "III"
    assert to_roman(2000) == "MM"


def test_rejects_out_of_range():
    with pytest.raises(ValueError):
        to_roman(0)


def test_subtractive_pairs():
    assert to_roman(4) == "IV"
    assert to_roman(9) == "IX"
    assert to_roman(90) == "XC"
    assert to_roman(2024) == "MMXXIV"
''',
        },
    },
    {
        "task_id": "chunked-drops-final-partial",
        "issue": (
            "chunked() silently discards the final chunk when the sequence length is "
            "not an exact multiple of the chunk size. chunked([1,2,3,4,5], 2) returns "
            "[[1,2],[3,4]] and loses the trailing [5]."
        ),
        "files": {
            "chunking.py": '''def chunked(items, size):
    """Split items into consecutive lists of at most `size` elements."""

    if size < 1:
        raise ValueError("size must be positive")
    chunks = []
    for start in range(0, len(items) - size + 1, size):
        chunks.append(items[start:start + size])
    return chunks
''',
            "test_chunking.py": '''import pytest

from chunking import chunked


def test_exact_multiple():
    assert chunked([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_empty_input():
    assert chunked([], 3) == []


def test_rejects_zero_size():
    with pytest.raises(ValueError):
        chunked([1, 2], 0)


def test_keeps_final_partial_chunk():
    assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
''',
        },
    },
    {
        "task_id": "binary-search-misses-last",
        "issue": (
            "binary_search() cannot find the final element of a range and returns -1 "
            "for a single-element list. binary_search([1,3,5,7], 7) returns -1 and "
            "binary_search([5], 5) returns -1."
        ),
        "files": {
            "search.py": '''def binary_search(sorted_values, target):
    """Return the index of target in a sorted list, or -1 when absent."""

    low = 0
    high = len(sorted_values) - 1
    while low < high:
        middle = (low + high) // 2
        if sorted_values[middle] == target:
            return middle
        if sorted_values[middle] < target:
            low = middle + 1
        else:
            high = middle - 1
    return -1
''',
            "test_search.py": '''from search import binary_search


def test_finds_interior_value():
    assert binary_search([1, 3, 5, 7], 3) == 1


def test_absent_value():
    assert binary_search([1, 3, 5, 7], 4) == -1


def test_empty_list():
    assert binary_search([], 1) == -1


def test_finds_final_and_single_values():
    assert binary_search([1, 3, 5, 7], 7) == 3
    assert binary_search([5], 5) == 0
''',
        },
    },
    {
        "task_id": "duration-ignores-components",
        "issue": (
            "parse_seconds() returns only the largest unit instead of the total. "
            "parse_seconds('1h30m') returns 3600 but should return 5400."
        ),
        "files": {
            "duration.py": '''import re

PATTERN = re.compile(r"^(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?$")


def parse_seconds(text):
    """Parse durations such as '90s', '5m' or '1h30m' into a number of seconds."""

    match = PATTERN.match(text.strip())
    if not match or not any(match.groups()):
        raise ValueError("unrecognised duration: %r" % text)
    hours, minutes, seconds = match.groups()
    if hours:
        return int(hours) * 3600
    if minutes:
        return int(minutes) * 60
    return int(seconds)
''',
            "test_duration.py": '''import pytest

from duration import parse_seconds


def test_single_units():
    assert parse_seconds("90s") == 90
    assert parse_seconds("5m") == 300
    assert parse_seconds("2h") == 7200


def test_rejects_garbage():
    with pytest.raises(ValueError):
        parse_seconds("later")


def test_combined_units_are_summed():
    assert parse_seconds("1h30m") == 5400
    assert parse_seconds("1h30m15s") == 5415
''',
        },
    },
    {
        "task_id": "csv-quoted-comma",
        "issue": (
            "split_row() splits on every comma, including commas that appear inside a "
            "double-quoted field. A row whose first field is the quoted value "
            "Doe, John and whose second field is 42 must parse as two fields, "
            "['Doe, John', '42'], but currently parses as three."
        ),
        "files": {
            "csvline.py": '''def split_row(line):
    """Split a single CSV row into fields, honouring double-quoted values."""

    return [field.strip('"') for field in line.split(",")]
''',
            "test_csvline.py": '''from csvline import split_row


def test_plain_fields():
    assert split_row("a,b,c") == ["a", "b", "c"]


def test_quoted_fields_without_commas():
    assert split_row('"a","b"') == ["a", "b"]


def test_comma_inside_quotes_is_not_a_separator():
    assert split_row('"Doe, John",42') == ["Doe, John", "42"]
''',
        },
    },
    {
        "task_id": "lru-evicts-wrong-entry",
        "issue": (
            "LRUCache evicts the most recently used key instead of the least recently "
            "used one. After filling a cache of capacity 2 with 'a' and 'b' and then "
            "inserting 'c', the cache drops 'b' and keeps 'a'."
        ),
        "files": {
            "lru.py": '''class LRUCache:
    """Fixed-capacity cache that evicts the least recently used key."""

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._entries = {}
        self._order = []

    def __len__(self):
        return len(self._entries)

    def get(self, key, default=None):
        if key not in self._entries:
            return default
        self._order.remove(key)
        self._order.append(key)
        return self._entries[key]

    def put(self, key, value):
        if key not in self._entries and len(self._entries) >= self.capacity:
            evicted = self._order.pop()
            del self._entries[evicted]
        if key in self._order:
            self._order.remove(key)
        self._entries[key] = value
        self._order.append(key)
''',
            "test_lru.py": '''import pytest

from lru import LRUCache


def test_stores_and_returns_values():
    cache = LRUCache(2)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("missing") is None


def test_rejects_zero_capacity():
    with pytest.raises(ValueError):
        LRUCache(0)


def test_evicts_the_least_recently_used_key():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert len(cache) == 2
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_reading_a_key_makes_it_recent():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert cache.get("a") == 1
    assert cache.get("b") is None
''',
        },
    },
    {
        "task_id": "wrap-ignores-separator-width",
        "issue": (
            "wrap() does not count the space it inserts between words, so a returned "
            "line can be longer than the requested width. wrap('the quick brown fox', 8) "
            "produces the 9-character line 'the quick'."
        ),
        "files": {
            "wrap.py": '''def wrap(text, width):
    """Wrap text at word boundaries into lines of at most `width` characters."""

    if width < 1:
        raise ValueError("width must be positive")
    lines = []
    current = ""
    for word in text.split():
        if current and len(current) + len(word) > width:
            lines.append(current)
            current = word
        elif current:
            current = current + " " + word
        else:
            current = word
    if current:
        lines.append(current)
    return lines
''',
            "test_wrap.py": '''import pytest

from wrap import wrap


def test_empty_text():
    assert wrap("", 5) == []


def test_rejects_zero_width():
    with pytest.raises(ValueError):
        wrap("a", 0)


def test_wraps_at_word_boundaries():
    assert wrap("the quick brown fox", 10) == ["the quick", "brown fox"]


def test_never_exceeds_the_requested_width():
    for width in (6, 7, 8, 9):
        for line in wrap("the quick brown fox jumps", width):
            assert len(line) <= width
''',
        },
    },
]


def git(arguments: List[str], cwd: Path) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_AUTHOR_NAME": "Benchmark",
            "GIT_AUTHOR_EMAIL": "benchmark@example.com",
            "GIT_COMMITTER_NAME": "Benchmark",
            "GIT_COMMITTER_EMAIL": "benchmark@example.com",
        }
    )
    return subprocess.run(
        ["git"] + arguments, cwd=str(cwd), env=environment,
        capture_output=True, text=True, check=True,
    )


def build_repository(task: Dict[str, Any], root: Path) -> str:
    path = root / task["task_id"]
    if path.exists():
        shutil.rmtree(str(path))
    path.mkdir(parents=True)
    for relative, content in task["files"].items():
        with (path / relative).open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
    git(["init", "--quiet"], path)
    git(["config", "core.autocrlf", "false"], path)
    git(["add", "-A"], path)
    git(["commit", "--quiet", "-m", "Seed {}".format(task["task_id"])], path)
    return git(["rev-parse", "HEAD"], path).stdout.strip()


def seeded_bug_is_caught(path: Path) -> bool:
    """A task is only meaningful if the suite fails before the agent touches it."""

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=str(path),
        capture_output=True, text=True, check=False,
    )
    return completed.returncode != 0


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repositories", type=Path, default=here / "local")
    parser.add_argument("--output", type=Path, default=here / "tasks.local.json")
    arguments = parser.parse_args()

    arguments.repositories.mkdir(parents=True, exist_ok=True)
    entries = []
    for task in TASKS:
        commit = build_repository(task, arguments.repositories)
        path = arguments.repositories / task["task_id"]
        if not seeded_bug_is_caught(path):
            raise SystemExit(
                "{}: the suite passes unmodified, so the task proves nothing".format(
                    task["task_id"]
                )
            )
        entries.append(
            {
                "task_id": task["task_id"],
                "repository_url": path.as_posix(),
                "base_commit": commit,
                "issue": task["issue"],
                "test_command": ["python", "-m", "pytest", "-q"],
            }
        )
        print("{:<34} {} (bug reproduced)".format(task["task_id"], commit[:10]))

    with arguments.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(entries, handle, indent=2)
        handle.write("\n")
    print("\nwrote {} tasks to {}".format(len(entries), arguments.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
