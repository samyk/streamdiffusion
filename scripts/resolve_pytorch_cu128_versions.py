#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
import urllib.error
import urllib.request
from html.parser import HTMLParser

INDEX = "https://download.pytorch.org/whl/nightly/cu128"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value:
                self.links.append(value)


def fetch_index_links(package: str) -> list[str]:
    url = f"{INDEX}/{package}/"
    request = urllib.request.Request(url, headers={"User-Agent": "streamdiffusion-td-bridge/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        html = response.read().decode("utf-8", errors="replace")
    parser = LinkParser()
    parser.feed(html)
    return parser.links


def versions_from_links(package: str, links: list[str], *, cp313: bool = True) -> list[str]:
    versions: list[str] = []
    pattern = re.compile(
        rf"{re.escape(package)}-([^/]+?)(?:%2Bcu128|\+cu128)-cp313"
        if cp313
        else rf"{re.escape(package)}-([^/]+?)(?:%2Bcu128|\+cu128)"
    )
    for link in links:
        match = pattern.search(link)
        if not match:
            continue
        version = match.group(1).replace("%2B", "+")
        if "+cu128" not in version:
            version = f"{version}+cu128"
        if version not in versions:
            versions.append(version)
    return versions


def pip_index_versions(package: str) -> list[str]:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "index", "versions", package, "--index-url", INDEX],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        return []

    versions: list[str] = []
    for line in out.splitlines():
        if "Available versions:" in line:
            _, _, tail = line.partition("Available versions:")
            versions.extend(version.strip() for version in tail.split(",") if version.strip())
        match = re.search(rf"^{re.escape(package)}\s+\(([^)]+)\)", line.strip())
        if match:
            versions.insert(0, match.group(1).strip())
    deduped: list[str] = []
    for version in versions:
        if version not in deduped:
            deduped.append(version)
    return deduped


def discover_versions(package: str) -> list[str]:
    versions = [version for version in pip_index_versions(package) if "+cu128" in version]
    if versions:
        return versions

    try:
        links = fetch_index_links(package)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"Could not fetch {package} index: {exc}") from exc

    versions = versions_from_links(package, links, cp313=True)
    if versions:
        return versions

    return versions_from_links(package, links, cp313=False)


def dev_date(version: str) -> str | None:
    match = re.search(r"dev(\d+)", version)
    return match.group(1) if match else None


def pick_same_date(package: str, date: str, versions: list[str]) -> str | None:
    for version in versions:
        if f"dev{date}" in version and "+cu128" in version:
            return version
    return None


def main() -> None:
    torch_versions = discover_versions("torch")
    tv_versions = discover_versions("torchvision")
    ta_versions = discover_versions("torchaudio")

    if not torch_versions:
        raise SystemExit(
            "No torch+cu128 wheels found on the nightly index. "
            "Check network access to download.pytorch.org."
        )

    torch_version = torch_versions[0]
    date = dev_date(torch_version)
    if not date:
        raise SystemExit(f"Could not parse dev date from torch version {torch_version}")

    tv_version = pick_same_date("torchvision", date, tv_versions) or (tv_versions[0] if tv_versions else "")
    ta_version = pick_same_date("torchaudio", date, ta_versions) or (ta_versions[0] if ta_versions else "")

    if not tv_version or not ta_version:
        raise SystemExit("Could not resolve torchvision/torchaudio cu128 wheels from the nightly index.")

    print(torch_version)
    print(tv_version)
    print(ta_version)


if __name__ == "__main__":
    main()
