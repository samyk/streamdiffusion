"""Apply frozen node positions from network_layout.py."""

from __future__ import annotations

import td

from network_layout import POSITIONS, PROJECT_ROOTS


def _op(path: str):
    """TD builtin `op` is not visible inside imported modules — use td.op."""
    return td.op(path)


def managed_roots(profile) -> list[str]:
    return [
        profile.hal_control,
        profile.hal_control_ui,
        profile.sync_dat,
        profile.parexec,
        profile.vidin,
        profile.vidout,
        profile.preview,
        profile.combine,
    ]


def _path_in_roots(path: str, roots: list[str]) -> bool:
    for root in roots:
        if path == root or path.startswith(f"{root}/"):
            return True
    return False


def collect_positions(roots: list[str]) -> dict[str, tuple[int, int]]:
    """Walk roots and COMP children; return absolute path -> (nodeX, nodeY)."""
    found: dict[str, tuple[int, int]] = {}

    def _visit(node) -> None:
        if node is None:
            return
        found[node.path] = (int(node.nodeX), int(node.nodeY))
        if hasattr(node, "children"):
            for child in node.children:
                _visit(child)

    for root in roots:
        _visit(_op(root))
    return found


def place(node) -> bool:
    """Set nodeX/nodeY from POSITIONS when this node's path is known."""
    if node is None:
        return False
    pos = POSITIONS.get(node.path)
    if pos is None:
        return False
    node.nodeX, node.nodeY = pos
    return True


def place_path(path: str) -> bool:
    return place(_op(path))


def apply_layout(profile, *, extra_roots: list[str] | None = None) -> int:
    """Place saved positions for this instance (no network walk)."""
    roots = list(managed_roots(profile))
    if extra_roots:
        roots.extend(extra_roots)
    placed = 0
    for path in POSITIONS:
        if not _path_in_roots(path, roots):
            continue
        if place_path(path):
            placed += 1
    return placed


def apply_project_layout() -> int:
    placed = 0
    for path in POSITIONS:
        if path in PROJECT_ROOTS or any(
            path.startswith(f"{root}/") for root in PROJECT_ROOTS
        ):
            if place_path(path):
                placed += 1
    return placed
