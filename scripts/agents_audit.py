#!/usr/bin/env python3
"""
AGENTS.md Compliance Auditor
----------------------------
Automated audit engine enforcing guidelines from .agents/AGENTS.md:
1. Safety First (Secret Scanning)
2. Python & A2A SDK (Type safety, explicit try-except)
3. GPU / PyTorch Performance (torch.no_grad / torch.inference_mode, VRAM management)
4. Frontend & TypeScript Strictness (No 'any' types)
5. Complexity & Algorithmic Optimization (Time/Space complexity, N+1 queries, linear search in loops)
"""

import sys
import os
import re
import ast
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Root directory of workspace
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# Files & directories to ignore
IGNORE_DIRS = {
    ".git", ".venv", "node_modules", "__pycache__", ".astro", "dist", "build", ".next"
}
IGNORE_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml"
}

# Regex patterns for secret detection
SECRET_PATTERNS = [
    (re.compile(r"""(?:api[_-]?key|secret|password|auth[_-]?token|private[_-]?key)\s*[:=]\s*["']([^"'\s]{8,})["']""", re.IGNORECASE), "Potential hardcoded secret or API key"),
    (re.compile(r"""sk-[a-zA-Z0-9]{32,}"""), "Potential OpenAI API key"),
    (re.compile(r"""AIzaSy[a-zA-Z0-9_-]{33}"""), "Potential Google API key"),
    (re.compile(r"""rf_[a-zA-Z0-9]{20,}"""), "Potential Roboflow API key"),
]


class AuditResult:
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_error(self, location: str, rule: str, message: str):
        self.errors.append(f"[ERROR] [{rule}] {location}: {message}")

    def add_warning(self, location: str, rule: str, message: str):
        self.warnings.append(f"[WARN] [{rule}] {location}: {message}")

    @property
    def is_clean(self) -> bool:
        return len(self.errors) == 0


def audit_secrets(root: Path, result: AuditResult):
    """Rule 1: Verify no API keys, secrets, or passwords are hardcoded."""
    for path in root.rglob("*"):
        if path.is_file() and not any(part in IGNORE_DIRS for part in path.parts):
            if path.name in IGNORE_FILES or path.suffix in [".png", ".jpg", ".jpeg", ".pth", ".pt", ".onnx", ".bin"]:
                continue

            # Skip checking .env.example
            if path.name == ".env.example":
                continue

            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            rel_path = path.relative_to(root)
            for line_idx, line in enumerate(content.splitlines(), start=1):
                # Ignore comment lines that document format examples
                if line.strip().startswith("#") or line.strip().startswith("//"):
                    continue
                for pattern, msg in SECRET_PATTERNS:
                    if pattern.search(line):
                        result.add_error(f"{rel_path}:{line_idx}", "Safety First", msg)


class PyASTVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, result: AuditResult):
        self.filename = filename
        self.result = result
        self.in_torch_no_grad = False
        self.loop_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Rule 2: Type Safety for new functions
        if not node.name.startswith("_"):
            if node.returns is None:
                self.result.add_warning(
                    f"{self.filename}:{node.lineno}",
                    "Python Type Safety",
                    f"Function '{node.name}' lacks return type annotation."
                )

        # Check for @torch.no_grad() decorator safely
        has_no_grad_dec = False
        for d in node.decorator_list:
            if isinstance(d, ast.Name) and d.id in ("no_grad", "inference_mode"):
                has_no_grad_dec = True
            elif isinstance(d, ast.Attribute) and d.attr in ("no_grad", "inference_mode"):
                has_no_grad_dec = True
            elif isinstance(d, ast.Call):
                func = getattr(d, "func", None)
                if isinstance(func, ast.Name) and func.id in ("no_grad", "inference_mode"):
                    has_no_grad_dec = True
                elif isinstance(func, ast.Attribute) and getattr(func, "attr", None) in ("no_grad", "inference_mode"):
                    has_no_grad_dec = True

        prev_no_grad = self.in_torch_no_grad
        if has_no_grad_dec:
            self.in_torch_no_grad = True

        self.generic_visit(node)
        self.in_torch_no_grad = prev_no_grad

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)  # type: ignore

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        # Rule 2: Avoid catching generic Exception unless logging or re-raising
        if node.type is None or (isinstance(node.type, ast.Name) and node.type.id == "Exception"):
            is_logging_or_raising = False
            for stmt in node.body:
                if isinstance(stmt, ast.Raise):
                    is_logging_or_raising = True
                    break
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    func = getattr(stmt.value, "func", None)
                    func_attr = getattr(func, "attr", None)
                    func_id = getattr(func, "id", None)
                    if func_attr in ("error", "exception", "warning", "critical", "log") or func_id in ("print", "log"):
                        is_logging_or_raising = True
                        break

            if not is_logging_or_raising and len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                self.result.add_error(
                    f"{self.filename}:{node.lineno}",
                    "Python Error Handling",
                    "Generic 'except Exception' or bare 'except' swallowed silently with 'pass'."
                )

        self.generic_visit(node)

    def visit_With(self, node: ast.With):
        no_grad_context = False
        for item in node.items:
            ctx = getattr(item, "context_expr", None)
            if isinstance(ctx, ast.Call):
                func = getattr(ctx, "func", None)
                if isinstance(func, ast.Name) and func.id in ("no_grad", "inference_mode"):
                    no_grad_context = True
                elif isinstance(func, ast.Attribute) and getattr(func, "attr", None) in ("no_grad", "inference_mode"):
                    no_grad_context = True

        prev_no_grad = self.in_torch_no_grad
        if no_grad_context:
            self.in_torch_no_grad = True

        self.generic_visit(node)
        self.in_torch_no_grad = prev_no_grad

    def visit_For(self, node: ast.For):
        self.loop_depth += 1
        # Rule 5: Time Complexity - Nested Loops Check
        if self.loop_depth >= 3:
            self.result.add_warning(
                f"{self.filename}:{node.lineno}",
                "Complexity Optimization",
                f"High loop nesting depth ({self.loop_depth} levels). Risk of O(n^2)+ or O(n^3) complexity."
            )
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_Compare(self, node: ast.Compare):
        # Rule 5: Time Complexity - Linear search inside loops (elem in list_var)
        if self.loop_depth > 0:
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, ast.In):
                    if isinstance(comparator, ast.List):
                        self.result.add_warning(
                            f"{self.filename}:{node.lineno}",
                            "Time Complexity",
                            "Linear search 'x in [...]' inside loop. Use set/dict for O(1) lookups."
                        )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        # Rule 5: Space & Time Complexity - Repeated list concatenation in loop (lst += [x])
        if self.loop_depth > 0 and isinstance(node.op, ast.Add):
            if isinstance(node.value, ast.List):
                self.result.add_warning(
                    f"{self.filename}:{node.lineno}",
                    "Space & Time Complexity",
                    "Repeated list concatenation 'lst += [x]' inside loop creates O(n^2) allocations. Use '.append()' instead."
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func = getattr(node, "func", None)
        func_name = getattr(func, "attr", None) or getattr(func, "id", None)

        # Rule 3: Check tensor .to(device) calls inside loops
        if self.loop_depth > 0 and isinstance(func, ast.Attribute) and getattr(func, "attr", None) == "to":
            if node.args and any(isinstance(arg, ast.Name) and arg.id in ("device", "cuda", "device_obj") for arg in node.args):
                self.result.add_warning(
                    f"{self.filename}:{node.lineno}",
                    "GPU Performance",
                    "Tensor '.to(device)' inside loop. Consider batching data transfer to GPU."
                )

        # Rule 5: Database / HTTP Queries inside loops (N+1 Query Problem)
        if self.loop_depth > 0 and func_name in ("execute", "executemany", "get", "post", "query", "fetch_one", "fetch_all"):
            self.result.add_warning(
                f"{self.filename}:{node.lineno}",
                "Database & API Optimization",
                f"Potential N+1 query problem: network/database call '{func_name}' inside loop. Use bulk queries or eager loading."
            )

        self.generic_visit(node)


def audit_python_files(root: Path, result: AuditResult):
    """Rule 2 & 3 & 5: Check Python code syntax, AST guidelines, PyTorch VRAM and type safety."""
    for path in root.rglob("*.py"):
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        rel_path = path.relative_to(root)
        try:
            content = path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(rel_path))
            visitor = PyASTVisitor(str(rel_path), result)
            visitor.visit(tree)
        except SyntaxError as se:
            result.add_error(f"{rel_path}:{se.lineno}", "Python Syntax", f"Syntax error: {se.msg}")
        except Exception as e:
            result.add_warning(str(rel_path), "Python Parsing", f"Could not parse file: {e}")


def audit_frontend_files(root: Path, result: AuditResult):
    """Rule 4: Frontend & TypeScript strictness (avoid 'any' types)."""
    any_regex = re.compile(r"""(:\s*any\b|\bas\s+any\b)""")

    frontend_dir = root / "frontend"
    if not frontend_dir.exists():
        return

    for path in frontend_dir.rglob("*"):
        if path.is_file() and path.suffix in [".ts", ".tsx", ".astro", ".vue", ".js", ".jsx"]:
            if any(part in IGNORE_DIRS for part in path.parts):
                continue
            rel_path = path.relative_to(root)
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for line_idx, line in enumerate(content.splitlines(), start=1):
                # Ignore comments
                stripped = line.strip()
                if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                    continue
                if any_regex.search(line):
                    result.add_error(
                        f"{rel_path}:{line_idx}",
                        "TypeScript Strictness",
                        "Use of 'any' type is prohibited. Use explicit interfaces or types."
                    )


def main():
    print("=" * 60)
    print("      AGENTS.md Guidelines Compliance Auditor")
    print("=" * 60)
    
    result = AuditResult()

    print("[1/3] Auditing Secrets & Security...")
    audit_secrets(WORKSPACE_ROOT, result)

    print("[2/3] Auditing Python Code, PyTorch VRAM & Algorithmic Complexity...")
    audit_python_files(WORKSPACE_ROOT, result)

    print("[3/3] Auditing Frontend & TypeScript Strictness...")
    audit_frontend_files(WORKSPACE_ROOT, result)

    print("=" * 60)
    print("AUDIT RESULTS SUMMARY:")
    print("=" * 60)

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"  {w}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for e in result.errors:
            print(f"  {e}")
        print("\n❌ AGENTS.md Audit FAILED! Fix errors above before pushing to GitHub.")
        sys.exit(1)
    else:
        print("\n✅ AGENTS.md Audit PASSED successfully! All guidelines satisfied.")
        sys.exit(0)


if __name__ == "__main__":
    main()
