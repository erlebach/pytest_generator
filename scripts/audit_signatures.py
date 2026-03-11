"""Audit return annotations and parameter naming for public checker functions.

Walks only module-level function definitions in assert_utilities.py.
Checks:
  - Return annotation must be "CheckResult" (not bare tuple[bool, str] or missing)
  - First param must be "student_answer"
  - For check_answer_* functions, second param must be "instructor_answer"

Exits 0 if no violations; exits 1 if any violations found.
"""

import ast
import sys
from pathlib import Path

TARGET = Path(__file__).parent.parent / "src" / "pytest_generator" / "assert_utilities.py"


def audit(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    violations: list[str] = []
    count = 0

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        name = node.name
        if not (name.startswith("check_structure_") or name.startswith("check_answer_")):
            continue

        count += 1
        lineno = node.lineno

        # Check return annotation
        if node.returns is None:
            violations.append(f"L{lineno}: {name} — missing return annotation")
        else:
            annotation = ast.unparse(node.returns)
            if annotation != "CheckResult":
                violations.append(
                    f"L{lineno}: {name} — return annotation is '{annotation}', expected 'CheckResult'"
                )

        # Check parameter naming
        args = node.args.args
        if not args:
            violations.append(f"L{lineno}: {name} — no parameters (expected at least student_answer)")
        else:
            if args[0].arg != "student_answer":
                violations.append(
                    f"L{lineno}: {name} — param[0] is '{args[0].arg}', expected 'student_answer'"
                )
            if name.startswith("check_answer_"):
                if len(args) < 2:
                    violations.append(
                        f"L{lineno}: {name} — fewer than 2 params (expected student_answer, instructor_answer)"
                    )
                elif args[1].arg != "instructor_answer":
                    violations.append(
                        f"L{lineno}: {name} — param[1] is '{args[1].arg}', expected 'instructor_answer'"
                    )

    return violations, count


def main() -> None:
    violations, count = audit(TARGET)
    if violations:
        print(f"VIOLATIONS ({len(violations)} found):")
        for v in violations:
            print(f"  {v}")
        sys.exit(1)
    else:
        print(f"OK: {count} check_structure_* / check_answer_* functions audited, 0 violations.")
        sys.exit(0)


if __name__ == "__main__":
    main()
