"""Instance profiles for parallel StreamDiffusion + NDI pipelines.

Connection defaults for instance A match streamdiffusion_td_bridge/defaults.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstanceProfile:
    key: str
    suffix: str
    label: str
    ndi_out: str
    ndi_in: str
    stream_id: str
    daydream_port: int
    control_port: int
    hal_host: str
    screen_session: str
    pidfile: str

    @property
    def vidin(self) -> str:
        return f"/project1/vidin{self.suffix}"

    @property
    def vidout(self) -> str:
        return f"/project1/vidout{self.suffix}"

    @property
    def hal_control(self) -> str:
        return f"/project1/hal_control{self.suffix}"

    @property
    def hal_control_ui(self) -> str:
        return f"/project1/hal_control_ui{self.suffix}"

    @property
    def sync_dat(self) -> str:
        return f"/project1/hal_remote_sync{self.suffix}"

    @property
    def parexec(self) -> str:
        return f"/project1/hal_remote_parexec{self.suffix}"

    @property
    def combine_layout(self) -> str:
        return f"{self.vidout}/combine_layout"

    @property
    def combine(self) -> str:
        return f"{self.vidout}/combine"

    @property
    def preview(self) -> str:
        return f"/project1/null1{self.suffix}"

    @property
    def layout_exec(self) -> str:
        return f"{self.vidout}/combine_layout_exec"


INSTANCES: dict[str, InstanceProfile] = {
    "a": InstanceProfile(
        key="a",
        suffix="",
        label="A",
        ndi_out="td_streamdiffusion_in",
        ndi_in="streamdiffusion_out",
        stream_id="remote-1",
        daydream_port=8780,
        control_port=8765,
        hal_host="192.168.0.90",
        screen_session="sdtd-bridge",
        pidfile="/tmp/sdtd-bridge.pid",
    ),
    "b": InstanceProfile(
        key="b",
        suffix="_b",
        label="B",
        ndi_out="td_streamdiffusion_in_b",
        ndi_in="streamdiffusion_out_b",
        stream_id="remote-2",
        daydream_port=8781,
        control_port=8766,
        hal_host="192.168.0.90",
        screen_session="sdtd-bridge-b",
        pidfile="/tmp/sdtd-bridge-b.pid",
    ),
}


def get_instance(key: str = "a") -> InstanceProfile:
    normalized = key.lower().strip()
    if normalized not in INSTANCES:
        known = ", ".join(sorted(INSTANCES))
        raise ValueError(f"Unknown instance {key!r}. Known: {known}")
    return INSTANCES[normalized]
