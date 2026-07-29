"""Shared helpers for one-shot TouchDesigner network build scripts (not runtime)."""

from __future__ import annotations

import td

from td_layout import place


def td_op(path: str):
    """TD textport `op` is not in scope for imported modules."""
    return td.op(path)


def has_par(target, par_name: str) -> bool:
    return hasattr(target.par, par_name)


def set_expr(target, par_name: str, expression: str) -> bool:
    if not has_par(target, par_name):
        return False
    getattr(target.par, par_name).expr = expression
    return True


def set_const(target, par_name: str, value) -> bool:
    if not has_par(target, par_name):
        return False
    par = getattr(target.par, par_name)
    par.expr = ""
    par.val = value
    return True


def set_first_match(target, par_names: tuple[str, ...], *, expr: str | None = None, val=None) -> str | None:
    for name in par_names:
        if not has_par(target, name):
            continue
        if expr is not None:
            set_expr(target, name, expr)
        elif val is not None:
            set_const(target, name, val)
        return name
    return None


def ensure(parent, name: str, op_type: str):
    node = parent.op(name)
    if node is not None and getattr(node, "OPType", "") != op_type:
        node.destroy()
        node = None
    if node is None:
        try:
            node = parent.create(op_type, name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not create {op_type} {name!r} under {parent.path}: {exc}") from exc
    return node


def destroy(parent, name: str) -> None:
    node = parent.op(name)
    if node is not None:
        node.destroy()


def connect(in0, out0) -> None:
    if in0 is None or out0 is None:
        return
    in0.inputConnectors[0].connect(out0)


def cook_validate(comp, label: str) -> None:
    try:
        comp.cook(force=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{label}: cook failed: {exc}") from exc
    if hasattr(comp, "errors"):
        err = comp.errors()
        if err:
            raise RuntimeError(f"{label}: {err}")


def require_no_errors(node, *, label: str) -> None:
    if node is None:
        raise RuntimeError(f"{label}: missing operator")
    if hasattr(node, "errors"):
        err = node.errors()
        if err:
            raise RuntimeError(f"{label}: {err}")


def require_chop(chop, *, label: str, min_chans: int = 1) -> None:
    if chop is None:
        raise RuntimeError(f"{label}: missing CHOP")
    try:
        count = chop.numChans
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{label}: cannot read numChans: {exc}") from exc
    if count < min_chans:
        raise RuntimeError(f"{label}: expected >= {min_chans} channels, got {count}")


def require_expr(target, par_name: str, *, label: str) -> None:
    if not has_par(target, par_name):
        raise RuntimeError(f"{label}: missing par {par_name}")
    expr = getattr(target.par, par_name).expr
    if not expr:
        raise RuntimeError(f"{label}: {par_name} has no expression")


def place_node(node) -> None:
    place(node)


def pulse_par(target, par_name: str) -> bool:
    if not has_par(target, par_name):
        return False
    getattr(target.par, par_name).pulse()
    return True
