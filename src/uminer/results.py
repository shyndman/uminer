from __future__ import annotations

import json
import os
import shutil
import zipfile
from functools import cached_property
from pathlib import Path
from typing import ClassVar, cast

from pydantic import BaseModel, ConfigDict

from .content import ContentList
from .errors import MinerUResultError
from .types import Json

CACHE_DIR_ENV = "XDG_CACHE_HOME"
EXTRACTED_DIR_NAME = "extracted"
LOCAL_OUTPUT_DIR_SUFFIX = ".uminer"
STAGING_DIR_SUFFIX = ".tmp"


class MinerUResultFile(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    path: str
    local_path: Path


class MinerUParsedResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    output_dir: Path
    zip_path: Path
    content_list: ContentList
    files: tuple[MinerUResultFile, ...]

    @cached_property
    def markdown(self) -> str | None:
        path = _find_file(self.files, "full.md")
        if path is None:
            return None
        return path.read_text(encoding="utf-8")

    @cached_property
    def raw_output(self) -> Json:
        path = _find_file_by_suffix(self.files, "_model.json")
        if path is None:
            return None
        return _read_json_file(path)

    @cached_property
    def layout(self) -> Json:
        path = _find_file(self.files, "layout.json") or _find_file_by_suffix(
            self.files, "_middle.json"
        )
        if path is None:
            return None
        return _read_json_file(path)

    @classmethod
    def from_zip_file(
        cls, zip_path: Path, output_dir: Path, *, extract_dir: Path | None = None
    ) -> MinerUParsedResult:
        extracted_dir = extract_dir or output_dir / EXTRACTED_DIR_NAME
        # Extract into a sibling staging dir, then atomically swap it into place
        # so an interrupted extraction never leaves a partial result directory.
        staging_dir = extracted_dir.with_name(extracted_dir.name + STAGING_DIR_SUFFIX)
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        _extract_zip(zip_path, staging_dir)
        if extracted_dir.exists():
            shutil.rmtree(extracted_dir)
        os.replace(staging_dir, extracted_dir)
        files = tuple(
            MinerUResultFile(
                path=path.relative_to(extracted_dir).as_posix(), local_path=path
            )
            for path in sorted(extracted_dir.rglob("*"))
            if path.is_file()
        )
        # MinerU also emits legacy content_list.json; this client ignores it.
        content_list_path = _find_file(
            files, "content_list_v2.json"
        ) or _find_file_by_suffix(files, "_content_list_v2.json")
        if content_list_path is None:
            raise MinerUResultError(
                "Expected content_list_v2.json in MinerU result zip"
            )
        return cls(
            output_dir=output_dir,
            zip_path=zip_path,
            content_list=ContentList.from_mineru(_read_json_file(content_list_path)),
            files=files,
        )


def default_result_cache_dir(result_id: str) -> Path:
    cache_home = os.getenv(CACHE_DIR_ENV)
    root = Path(cache_home) if cache_home else Path.home() / ".cache"
    return root / "uminer" / "results" / result_id


def default_local_output_dir(source_path: Path) -> Path:
    return source_path.with_name(source_path.name + LOCAL_OUTPUT_DIR_SUFFIX)


# Manual extraction (vs. ZipFile.extractall) to guard against zip-slip:
# the archive comes from a remote API, so member names are untrusted and
# could contain `../` or absolute paths that escape output_dir.
def _extract_zip(zip_path: Path, output_dir: Path) -> None:
    root = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in [m for m in archive.infolist() if not m.is_dir()]:
            target = (output_dir / member.filename).resolve()
            if not target.is_relative_to(root):
                raise MinerUResultError(
                    f"Unsafe path in MinerU result zip: {member.filename}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def _find_file(files: tuple[MinerUResultFile, ...], name: str) -> Path | None:
    for file in files:
        if file.path == name or file.path.endswith(f"/{name}"):
            return file.local_path
    return None


def _find_file_by_suffix(
    files: tuple[MinerUResultFile, ...], suffix: str
) -> Path | None:
    matches = [file.local_path for file in files if file.path.endswith(suffix)]
    if not matches:
        return None
    if len(matches) > 1:
        raise MinerUResultError(f"Expected one {suffix} file, found {len(matches)}")
    return matches[0]


def _read_json_file(path: Path) -> Json:
    return cast(Json, json.loads(path.read_text(encoding="utf-8")))
