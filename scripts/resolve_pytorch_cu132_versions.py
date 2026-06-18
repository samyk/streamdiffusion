#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
import urllib.error
import urllib.request
from html.parser import HTMLParser

INDEX = "https://download.pytorch.org/whl/nightly/cu132"
CUDA_TAG = "cu132"


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
        rf"{re.escape(package)}-([^/]+?)(?:%2B{CUDA_TAG}|\+{CUDA_TAG})-cp313"
        if cp313
        else rf"{re.escape(package)}-([^/]+?)(?:%2B{CUDA_TAG}|\+{CUDA_TAG})"
    )
    for link in links:
        match = pattern.search(link)
        if not match:
            continue
        version = match.group(1).replace("%2B", "+")
        if f"+{CUDA_TAG}" not in version:
            version = f"{version}+{CUDA_TAG}"
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
    versions = [version for version in pip_index_versions(package) if f"+{CUDA_TAG}" in version]
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


def by_dev_date(versions: list[str]) -> dict[str, str]:
    dated: dict[str, str] = {}
    for version in versions:
        date = dev_date(version)
        if date and date not in dated:
            dated[date] = version
    return dated


def pick_same_date(date: str, versions: list[str]) -> str | None:
    for version in versions:
        if f"dev{date}" in version and f"+{CUDA_TAG}" in version:
            return version
    return None


def resolve_triplet(
    torch_versions: list[str],
    tv_versions: list[str],
    ta_versions: list[str],
) -> tuple[str, str, str] | None:
    torch_dates = by_dev_date(torch_versions)
    ta_dates = by_dev_date(ta_versions)

    # torchvision declares an exact torch== dev build; anchor on torchvision dates.
    for tv in tv_versions:
        date = dev_date(tv)
        if not date:
            continue
        torch_version = torch_dates.get(date)
        if not torch_version:
            continue
        ta_version = ta_dates.get(date) or pick_same_date(date, ta_versions)
        if ta_version:
            return torch_version, tv, ta_version

    # Fallback: newest date where all three wheels exist.
    common_dates = sorted(
        set(torch_dates) & set(by_dev_date(tv_versions)) & set(ta_dates),
        reverse=True,
    )
    for date in common_dates:
        tv_version = pick_same_date(date, tv_versions)
        if tv_version:
            return torch_dates[date], tv_version, ta_dates[date]

    return None


def main() -> None:
    torch_versions = discover_versions("torch")
    tv_versions = discover_versions("torchvision")
    ta_versions = discover_versions("torchaudio")

    if not torch_versions or not tv_versions or not ta_versions:
        raise SystemExit(
            f"Missing torch/torchvision/torchaudio +{CUDA_TAG} wheels on the nightly index."
        )

    triplet = resolve_triplet(torch_versions, tv_versions, ta_versions)
    if not triplet:
        raise SystemExit(
            f"Could not find aligned dev-dated {CUDA_TAG} wheels. "
            "Retry later or pin versions manually."
        )

    torch_version, tv_version, ta_version = triplet
    print(torch_version)
    print(tv_version)
    print(ta_version)


if __name__ == "__main__":
    main()
