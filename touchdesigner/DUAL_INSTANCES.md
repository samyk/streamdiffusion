# Dual StreamDiffusion Instances

Run two independent pipelines in parallel: two TouchDesigner projects (or one TD with two trees), two bridge processes on `hal`, separate NDI names and REST stream IDs.

## Instance map

| | Instance A | Instance B |
|---|---|---|
| NDI send (TD → hal) | `td_streamdiffusion_in` | `td_streamdiffusion_in_b` |
| NDI return (hal → TD) | `streamdiffusion_out` | `streamdiffusion_out_b` |
| REST stream ID | `remote-1` | `remote-2` |
| Daydream API port | `8780` | `8781` |
| WebSocket port | `8765` | `8766` |
| HAL screen session | `sdtd-bridge` | `sdtd-bridge-b` |
| TD vidin | `/project1/vidin` | `/project1/vidin_b` |
| TD vidout | `/project1/vidout` | `/project1/vidout_b` |
| TD control | `/project1/hal_control` | `/project1/hal_control_b` |
| TD UI panel | `/project1/hal_control_ui` | `/project1/hal_control_ui_b` |
| TD preview | `/project1/null1` | `/project1/null1_b` |

## HAL — start both bridges

On `hal`:

```bash
cd /d/c/samysd

# Both instances (opt-in)
./scripts/run_bridge_screen.sh --dual "prompt A" "prompt B"

# Instance A only (default)
./scripts/run_bridge_screen.sh
./scripts/run_bridge_screen.sh "prompt A"

# Or individually
./scripts/run_bridge_instance.sh a "prompt for stream A"
./scripts/run_bridge_instance.sh b "prompt for stream B"
```

Attach logs:

```bash
screen -r sdtd-bridge      # instance A
screen -r sdtd-bridge-b    # instance B
```

Verify APIs:

```bash
curl http://192.168.0.90:8780/v1/streams/remote-1
curl http://192.168.0.90:8781/v1/streams/remote-2
```

## TouchDesigner — per machine / per project

**Recommended:** one TD project per live stream (two Macs or two TD apps). Each project runs:

```python
# Instance A project
exec(open("/Users/samy/c/touch/samysd/touchdesigner/build_instance.py", encoding="utf-8").read())
```

```python
# Instance B project (or second TD)
INSTANCE = "b"
exec(open("/Users/samy/c/touch/samysd/touchdesigner/build_instance.py", encoding="utf-8").read())
```

`build_instance.py` will:
1. Clone `vidin` → `vidin_b` and `vidout` → `vidout_b` (instance B only)
2. Wire NDI with instance-specific names
3. Create `hal_control[_b]` + UI panel
4. Build combine layout (PiP + text)

**Wire your camera/source** into each project's `vidin` (or `vidin_b`) separately.

**Open UI:** drag `/project1/hal_control_ui` (or `hal_control_ui_b`) into a Panel pane.

## Two TDs on one Mac

Works if each TD process sends a **unique NDI out name**. Run instance A in TD project 1, instance B in TD project 2 — each with matching `INSTANCE` build.

Do **not** run two `ndiout1` with the same NDI name in the same TD process.

**Use `TD_LAYOUT` so each TD app only controls one stream:**

```python
# TD app 1 — instance A only (do NOT keep hal_control_b here)
INSTANCE = "a"
TD_LAYOUT = "a_only"
exec(open(".../build_instance.py", encoding="utf-8").read())

# TD app 2 — instance B only (do NOT keep hal_control here)
INSTANCE = "b"
TD_LAYOUT = "b_only"
exec(open(".../build_instance.py", encoding="utf-8").read())
```

### Troubleshooting: `streamdiffusion_out` flickers between two prompts

This means **two clients are fighting over `remote-1` (:8780)** — not an NDI `_b` name collision.

Check Textport for lines like:
```
[hal_remote_sync] /project1/hal_control -> http://192.168.0.90:8780/v1/streams/remote-1 ...
[hal_remote_sync] /project1/hal_control -> ...   # from a second TD project
```

Fix:
1. Only **one** TD project may have `hal_control` → `remote-1` / port `8780`
2. The other TD must use `TD_LAYOUT = "b_only"` (`hal_control_b` → `remote-2` / `8781`)
3. Re-run `build_instance.py` on both projects after changing layout
4. Only one `vidin` should publish `td_streamdiffusion_in` (check NDI Studio)

Also verify `vidout/ndiin2` is `HAL (streamdiffusion_out)` — not `_out_b`.

## VRAM note

Two SDXL streams on one GPU doubles inference load. Consider `512×512` for B, or stagger presets, if VRAM is tight.

## Customize profiles

Edit `touchdesigner/instances.py` to change names, ports, or add instance `c`.
