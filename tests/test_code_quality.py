"""Static checks on the source code: no dead code, no unused names, clean structure.

Only the standard library (ast) is used, so the checks run everywhere the
unit tests run. Run from the project folder with:  python -m unittest -v
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "smith_waterman"
PRODUCTION = sorted(PACKAGE.glob("*.py")) + [ROOT / name for name in ("main.py", "gui.py", "demo.py", "benchmark.py")]
TESTS = sorted((ROOT / "tests").glob("*.py"))

# Modules that talk to the user and may print; the library itself only returns strings.
MAY_PRINT = {"cli.py", "main.py", "demo.py", "benchmark.py"}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


TREES = {path: parse(path) for path in PRODUCTION}


def loaded_names(tree: ast.AST) -> set[str]:
    """Every identifier read in the tree, as a bare name or as an attribute."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            names.add(node.attr)
    return names


def public_api() -> set[str]:
    """Names exported by the package through ``__all__``."""
    tree = TREES[PACKAGE / "__init__.py"]
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in node.targets):
            return set(ast.literal_eval(node.value))
    return set()


def where(path: Path, node: ast.AST) -> str:
    return f"{path.relative_to(ROOT)}:{node.lineno}"


class DeadCode(unittest.TestCase):
    """Every definition in the production code is used by the production code."""

    def setUp(self):
        self.used = set().union(*(loaded_names(tree) for tree in TREES.values()))
        self.api = public_api()

    def test_every_top_level_definition_is_used(self):
        unused = []
        for path, tree in TREES.items():
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    names = [node.name]
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    names = [t.id for t in targets if isinstance(t, ast.Name)]
                else:
                    continue
                for name in names:
                    if name == "__all__" or name in self.api:
                        continue
                    if name not in self.used:
                        unused.append(f"{where(path, node)} {name}")
        self.assertEqual(unused, [], "definitions nobody uses")

    def test_every_method_and_field_is_used(self):
        unused = []
        for path, tree in TREES.items():
            for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
                for node in cls.body:
                    if isinstance(node, ast.FunctionDef):
                        name = node.name
                    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                        name = node.target.id         # dataclass field
                    else:
                        continue
                    if name.startswith("__") and name.endswith("__"):
                        continue                      # dunder methods are called by Python itself
                    if name not in self.used:
                        unused.append(f"{where(path, node)} {cls.name}.{name}")
        self.assertEqual(unused, [], "methods / fields nobody reads")

    def test_no_unused_imports(self):
        unused = []
        for path, tree in TREES.items():
            used = loaded_names(tree)
            exported = public_api() if path.name == "__init__.py" else set()
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    continue
                for alias in node.names:
                    name = (alias.asname or alias.name).split(".")[0]
                    if name not in used and name not in exported:
                        unused.append(f"{where(path, node)} {name}")
        self.assertEqual(unused, [], "imports that are never used")

    def test_no_unused_local_variables_or_parameters(self):
        problems = []
        for path, tree in TREES.items():
            for func in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.Lambda))):
                body = func.body if isinstance(func.body, list) else [func.body]
                read = set().union(*(loaded_names(stmt) for stmt in body))
                args = func.args.posonlyargs + func.args.args + func.args.kwonlyargs
                for arg in args:
                    if arg.arg not in read and arg.arg not in ("self", "cls") and not arg.arg.startswith("_"):
                        problems.append(f"{where(path, func)} parameter '{arg.arg}' of {getattr(func, 'name', 'lambda')}")
                for node in (n for stmt in body for n in ast.walk(stmt)):
                    if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                            and not node.id.startswith("_") and node.id not in read):
                        problems.append(f"{where(path, node)} variable '{node.id}'")
        self.assertEqual(problems, [], "values that are computed and never read")

    def test_no_unreachable_statements(self):
        problems = []
        for path, tree in TREES.items():
            for node in ast.walk(tree):
                for field in ("body", "orelse", "finalbody"):
                    block = getattr(node, field, None)
                    if not isinstance(block, list):
                        continue
                    for stmt, following in zip(block, block[1:]):
                        if isinstance(stmt, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
                            problems.append(f"{where(path, following)} follows a {type(stmt).__name__.lower()}")
        self.assertEqual(problems, [], "code that can never run")

    def test_private_helpers_are_used_in_their_own_module(self):
        unused = []
        for path, tree in TREES.items():
            used = loaded_names(tree)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith("_") and node.name not in used:
                    unused.append(f"{where(path, node)} {node.name}")
        self.assertEqual(unused, [])


class Style(unittest.TestCase):
    """Conventions the project follows."""

    def test_no_bare_except(self):
        bare = [where(p, n) for p, t in TREES.items() for n in ast.walk(t)
                if isinstance(n, ast.ExceptHandler) and n.type is None]
        self.assertEqual(bare, [])

    def test_library_does_not_print(self):
        printing = [
            where(p, n) for p, t in TREES.items() if p.name not in MAY_PRINT
            for n in ast.walk(t)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"
        ]
        self.assertEqual(printing, [], "only the command line front ends may print")

    def test_every_module_and_public_function_is_documented(self):
        missing = []
        for path, tree in TREES.items():
            if not ast.get_docstring(tree):
                missing.append(f"{path.relative_to(ROOT)} (module)")
        for path in sorted(PACKAGE.glob("*.py")):
            for node in TREES[path].body:
                if isinstance(node, ast.ClassDef) and not ast.get_docstring(node):
                    missing.append(f"{where(path, node)} class {node.name}")
        self.assertEqual(missing, [])

    def test_no_leftover_markers(self):
        pattern = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b|breakpoint\(|pdb\.set_trace")
        found = [
            f"{path.relative_to(ROOT)}:{number}"
            for path in PRODUCTION + TESTS
            if path.name != Path(__file__).name
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if pattern.search(line)
        ]
        self.assertEqual(found, [])

    def test_public_api_names_exist(self):
        import smith_waterman

        for name in public_api():
            self.assertTrue(hasattr(smith_waterman, name), name)

    def test_source_is_ascii_or_utf8_without_tabs(self):
        for path in PRODUCTION + TESTS:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("\t", text, path.name)
            self.assertFalse(any(line != line.rstrip() for line in text.splitlines()), f"trailing spaces in {path.name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
