"""Fetch locked public source for baseline reproduction; no git submodules."""

import io
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    for part, item in lock.items():
        name = item["repository"].rsplit("/", 1)[-1]
        url = f"https://codeload.github.com/ahmedsohail2003/{name}/tar.gz/{item['revision']}"
        with urllib.request.urlopen(url, timeout=120) as response:
            archive = response.read()
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                relative = Path(*Path(member.name).parts[1:])
                if ".." in relative.parts:
                    raise ValueError(member.name)
                dest = ROOT / ".cache/upstream" / part / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tar.extractfile(member).read())
        print(part, item["revision"], flush=True)


if __name__ == "__main__":
    main()
