from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import cast, final, override
from uuid import UUID

import httpx
from pydantic import PositiveInt, validate_call

from .errors import MinerUApiError, MinerUConfigError
from .job import ExtractionBatch, ExtractionBatchItemResult, ExtractionJob
from .models import (
    BatchExtractResult,
    ExtractionSource,
    ExtractionStatus,
    ExtractTask,
    TaskPage,
    UploadBatch,
)
from .results import MinerUParsedResult, default_result_cache_dir
from .types import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    MODEL_VERSION,
    ExtraFormat,
    FileSpec,
    Json,
)

type UploadProgressCallback = Callable[[Path, int, int], None]
type BatchUploadProgressCallback = Callable[[int, Path, int, int], None]

UPLOAD_CHUNK_SIZE = 64 * 1024
UPLOAD_WRITE_TIMEOUT_SECONDS = 120.0
DEFAULT_BATCH_UPLOAD_WORKERS = 4


@final
class _ProgressByteStream(httpx.SyncByteStream):
    _path: Path
    _total_bytes: int
    _callback: UploadProgressCallback
    _uploaded_bytes: int

    def __init__(
        self, path: Path, total_bytes: int, callback: UploadProgressCallback
    ) -> None:
        self._path = path
        self._total_bytes = total_bytes
        self._callback = callback
        self._uploaded_bytes = 0

    @override
    def __iter__(self) -> Iterator[bytes]:
        with self._path.open("rb") as file:
            while True:
                chunk = file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                self._uploaded_bytes += len(chunk)
                self._callback(self._path, self._uploaded_bytes, self._total_bytes)
                yield chunk

    @override
    def close(self) -> None:
        return None


class MinerUClient:
    api_key: str
    _owns_client: bool
    _client: httpx.Client

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float | httpx.Timeout = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv(DEFAULT_API_KEY_ENV)
        if not resolved_api_key:
            raise MinerUConfigError(
                f"MinerU API key required. Pass api_key or set {DEFAULT_API_KEY_ENV}."
            )
        self.api_key = resolved_api_key
        self._owns_client = client is None
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> MinerUClient:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        _ = exc_type, exc_value, traceback
        self.close()

    def extract_url(
        self,
        url: str,
        *,
        is_ocr: bool | None = None,
        enable_formula: bool | None = None,
        enable_table: bool | None = None,
        language: str | None = None,
        data_id: str | None = None,
        extra_formats: Iterable[ExtraFormat] | None = None,
        page_ranges: str | None = None,
        no_cache: bool | None = None,
        cache_tolerance: int | None = None,
        poll_interval_seconds: float = 2.0,
    ) -> ExtractionJob:
        task = self.create_extract_task(
            url,
            is_ocr=is_ocr,
            enable_formula=enable_formula,
            enable_table=enable_table,
            language=language,
            data_id=data_id,
            extra_formats=extra_formats,
            page_ranges=page_ranges,
            no_cache=no_cache,
            cache_tolerance=cache_tolerance,
        )
        return ExtractionJob.from_task(
            self,
            task.task_id,
            source=ExtractionSource(kind="url", url=url),
            status=ExtractionStatus.from_task(task),
            poll_interval_seconds=poll_interval_seconds,
        )

    def extract_file(
        self,
        path: str | Path,
        *,
        file: FileSpec | None = None,
        enable_formula: bool | None = None,
        enable_table: bool | None = None,
        language: str | None = None,
        extra_formats: Iterable[ExtraFormat] | None = None,
        on_upload_progress: UploadProgressCallback | None = None,
        poll_interval_seconds: float = 2.0,
    ) -> ExtractionJob:
        file_path = Path(path)
        file_spec = file if file is not None else {"name": file_path.name}
        batch = self.create_file_upload_extract_tasks(
            [file_path],
            files=[file_spec],
            enable_formula=enable_formula,
            enable_table=enable_table,
            language=language,
            extra_formats=extra_formats,
            on_upload_progress=on_upload_progress,
        )
        return ExtractionJob.from_batch(
            self,
            batch.batch_id,
            source=ExtractionSource(kind="file", path=file_path, file=file_spec),
            status=ExtractionStatus(
                batch_id=batch.batch_id,
                state="waiting-file",
                file_name=str(file_spec.get("name", file_path.name)),
            ),
            poll_interval_seconds=poll_interval_seconds,
        )

    def extract_urls(
        self,
        urls: Iterable[str],
        *,
        is_ocr: bool | None = None,
        enable_formula: bool | None = None,
        enable_table: bool | None = None,
        language: str | None = None,
        extra_formats: Iterable[ExtraFormat] | None = None,
        page_ranges: str | None = None,
        no_cache: bool | None = None,
        cache_tolerance: int | None = None,
        poll_interval_seconds: float = 2.0,
    ) -> ExtractionBatch:
        del is_ocr, page_ranges
        url_tuple = tuple(urls)
        if not url_tuple:
            raise ValueError("Expected at least one URL")
        item_data_ids = tuple(
            _batch_item_data_id(index) for index in range(len(url_tuple))
        )
        batch_id = self.create_url_batch(
            [
                {"url": url, "data_id": data_id}
                for url, data_id in zip(url_tuple, item_data_ids, strict=True)
            ],
            enable_formula=enable_formula,
            enable_table=enable_table,
            language=language,
            extra_formats=extra_formats,
            no_cache=no_cache,
            cache_tolerance=cache_tolerance,
        )
        sources = tuple(ExtractionSource(kind="url", url=url) for url in url_tuple)
        statuses = tuple(
            ExtractionStatus(batch_id=batch_id, state=None, data_id=data_id)
            for data_id in item_data_ids
        )
        return ExtractionBatch.from_batch(
            self,
            batch_id,
            sources=sources,
            statuses=statuses,
            item_data_ids=item_data_ids,
            poll_interval_seconds=poll_interval_seconds,
        )

    def extract_files(
        self,
        paths: Iterable[str | Path],
        *,
        enable_formula: bool | None = None,
        enable_table: bool | None = None,
        language: str | None = None,
        extra_formats: Iterable[ExtraFormat] | None = None,
        on_upload_progress: BatchUploadProgressCallback | None = None,
        poll_interval_seconds: float = 2.0,
        max_upload_workers: int = DEFAULT_BATCH_UPLOAD_WORKERS,
    ) -> ExtractionBatch:
        path_tuple = tuple(Path(path) for path in paths)
        if not path_tuple:
            raise ValueError("Expected at least one file path")
        item_data_ids = tuple(
            _batch_item_data_id(index) for index in range(len(path_tuple))
        )
        file_specs = [
            {"name": path.name, "data_id": data_id}
            for path, data_id in zip(path_tuple, item_data_ids, strict=True)
        ]
        batch = self.create_upload_batch(
            file_specs,
            enable_formula=enable_formula,
            enable_table=enable_table,
            language=language,
            extra_formats=extra_formats,
        )
        upload_errors = self._upload_batch_files(
            tuple(batch.file_urls),
            path_tuple,
            on_progress=on_upload_progress,
            max_upload_workers=max_upload_workers,
        )
        sources = tuple(
            ExtractionSource(kind="file", path=path, file=file_spec)
            for path, file_spec in zip(path_tuple, file_specs, strict=True)
        )
        statuses: list[ExtractionStatus] = []
        results: list[ExtractionBatchItemResult | None] = []
        for index, source in enumerate(sources):
            file_name = file_specs[index]["name"]
            error = upload_errors.get(index)
            if error is None:
                statuses.append(
                    ExtractionStatus(
                        batch_id=batch.batch_id,
                        state="waiting-file",
                        file_name=file_name,
                        data_id=item_data_ids[index],
                    )
                )
                results.append(None)
                continue
            status = ExtractionStatus(
                batch_id=batch.batch_id,
                state="failed",
                file_name=file_name,
                data_id=item_data_ids[index],
                err_msg=str(error),
            )
            statuses.append(status)
            results.append(
                ExtractionBatchItemResult(source=source, status=status, error=error)
            )
        return ExtractionBatch.from_batch(
            self,
            batch.batch_id,
            sources=sources,
            statuses=tuple(statuses),
            item_data_ids=item_data_ids,
            poll_interval_seconds=poll_interval_seconds,
            results=tuple(results),
        )

    def create_extract_task(
        self,
        url: str,
        *,
        is_ocr: bool | None = None,
        enable_formula: bool | None = None,
        enable_table: bool | None = None,
        language: str | None = None,
        data_id: str | None = None,
        callback: str | None = None,
        seed: str | None = None,
        extra_formats: Iterable[ExtraFormat] | None = None,
        page_ranges: str | None = None,
        no_cache: bool | None = None,
        cache_tolerance: int | None = None,
    ) -> ExtractTask:
        data = self._request(
            "POST",
            "/api/v4/extract/task",
            json_body=_compact(
                {
                    "url": url,
                    "model_version": MODEL_VERSION,
                    "is_ocr": is_ocr,
                    "enable_formula": enable_formula,
                    "enable_table": enable_table,
                    "language": language,
                    "data_id": data_id,
                    "callback": callback,
                    "seed": seed,
                    "extra_formats": list(extra_formats)
                    if extra_formats is not None
                    else None,
                    "page_ranges": page_ranges,
                    "no_cache": no_cache,
                    "cache_tolerance": cache_tolerance,
                }
            ),
        )
        return ExtractTask.model_validate(data)

    def get_extract_task(self, task_id: str | UUID) -> ExtractTask:
        data = self._request("GET", f"/api/v4/extract/task/{task_id!s}")
        return ExtractTask.model_validate(data)

    @validate_call
    def list_tasks(
        self, *, page_no: PositiveInt = 1, page_size: PositiveInt = 20
    ) -> TaskPage:
        data = self._request(
            "GET", f"/api/v4/tasks?page_no={page_no}&page_size={page_size}"
        )
        return TaskPage.model_validate(data)

    def create_upload_batch(
        self,
        files: Iterable[FileSpec],
        *,
        enable_formula: bool | None = None,
        enable_table: bool | None = None,
        language: str | None = None,
        callback: str | None = None,
        seed: str | None = None,
        extra_formats: Iterable[ExtraFormat] | None = None,
    ) -> UploadBatch:
        data = self._request(
            "POST",
            "/api/v4/file-urls/batch",
            json_body=_compact(
                {
                    "files": list(files),
                    "model_version": MODEL_VERSION,
                    "enable_formula": enable_formula,
                    "enable_table": enable_table,
                    "language": language,
                    "callback": callback,
                    "seed": seed,
                    "extra_formats": list(extra_formats)
                    if extra_formats is not None
                    else None,
                }
            ),
        )
        return UploadBatch.model_validate(data)

    def upload_file(
        self,
        upload_url: str,
        path: str | Path,
        *,
        on_progress: UploadProgressCallback | None = None,
    ) -> None:
        path_obj = Path(path)
        base_timeout = self._client.timeout
        upload_timeout = httpx.Timeout(
            connect=base_timeout.connect,
            read=base_timeout.read,
            write=UPLOAD_WRITE_TIMEOUT_SECONDS,
            pool=base_timeout.pool,
        )
        if on_progress is None:
            with path_obj.open("rb") as file:
                response = self._client.put(
                    upload_url, content=file, timeout=upload_timeout
                )
        else:
            total_bytes = path_obj.stat().st_size
            response = self._client.put(
                upload_url,
                content=_ProgressByteStream(path_obj, total_bytes, on_progress),
                headers={"Content-Length": str(total_bytes)},
                timeout=upload_timeout,
            )
        _ = response.raise_for_status()

    def upload_files(
        self,
        upload_urls: Iterable[str],
        paths: Iterable[str | Path],
        *,
        on_progress: UploadProgressCallback | None = None,
    ) -> None:
        for upload_url, path in zip(upload_urls, paths, strict=True):
            self.upload_file(upload_url, path, on_progress=on_progress)

    def create_file_upload_extract_tasks(
        self,
        paths: Iterable[str | Path],
        *,
        files: Iterable[FileSpec] | None = None,
        enable_formula: bool | None = None,
        enable_table: bool | None = None,
        language: str | None = None,
        callback: str | None = None,
        seed: str | None = None,
        extra_formats: Iterable[ExtraFormat] | None = None,
        on_upload_progress: UploadProgressCallback | None = None,
    ) -> UploadBatch:
        path_tuple = tuple(Path(path) for path in paths)
        file_specs = (
            list(files)
            if files is not None
            else [{"name": path.name} for path in path_tuple]
        )
        batch = self.create_upload_batch(
            file_specs,
            enable_formula=enable_formula,
            enable_table=enable_table,
            language=language,
            callback=callback,
            seed=seed,
            extra_formats=extra_formats,
        )
        self.upload_files(
            batch.file_urls,
            path_tuple,
            on_progress=on_upload_progress,
        )
        return batch

    def _upload_batch_files(
        self,
        upload_urls: tuple[str, ...],
        paths: tuple[Path, ...],
        *,
        on_progress: BatchUploadProgressCallback | None,
        max_upload_workers: int,
    ) -> dict[int, Exception]:
        if len(upload_urls) != len(paths):
            raise ValueError("Expected one upload URL per path")
        if max_upload_workers < 1:
            raise ValueError("max_upload_workers must be at least 1")

        def upload_one(index: int, upload_url: str, path: Path) -> None:
            progress_callback = on_progress
            if progress_callback is not None:

                def on_file_progress(
                    current_path: Path, uploaded: int, total: int
                ) -> None:
                    progress_callback(index, current_path, uploaded, total)

                self.upload_file(upload_url, path, on_progress=on_file_progress)
                return
            self.upload_file(upload_url, path)

        errors: dict[int, Exception] = {}
        worker_count = min(max_upload_workers, len(paths))
        if worker_count == 1:
            for index, (upload_url, path) in enumerate(
                zip(upload_urls, paths, strict=True)
            ):
                try:
                    upload_one(index, upload_url, path)
                except Exception as exc:
                    errors[index] = exc
            return errors

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(upload_one, index, upload_url, path): index
                for index, (upload_url, path) in enumerate(
                    zip(upload_urls, paths, strict=True)
                )
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors[futures[future]] = exc
        return errors

    def create_url_batch(
        self,
        files: Iterable[FileSpec],
        *,
        enable_formula: bool | None = None,
        enable_table: bool | None = None,
        language: str | None = None,
        callback: str | None = None,
        seed: str | None = None,
        extra_formats: Iterable[ExtraFormat] | None = None,
        no_cache: bool | None = None,
        cache_tolerance: int | None = None,
    ) -> str:
        data = self._request(
            "POST",
            "/api/v4/extract/task/batch",
            json_body=_compact(
                {
                    "files": list(files),
                    "model_version": MODEL_VERSION,
                    "enable_formula": enable_formula,
                    "enable_table": enable_table,
                    "language": language,
                    "callback": callback,
                    "seed": seed,
                    "extra_formats": list(extra_formats)
                    if extra_formats is not None
                    else None,
                    "no_cache": no_cache,
                    "cache_tolerance": cache_tolerance,
                }
            ),
        )
        batch_id = data["batch_id"]
        if not isinstance(batch_id, str):
            raise TypeError("Expected batch_id to be str")
        return batch_id

    def get_batch_extract_result(self, batch_id: str) -> BatchExtractResult:
        data = self._request("GET", f"/api/v4/extract-results/batch/{batch_id}")
        return BatchExtractResult.model_validate(data)

    def download_result(
        self,
        full_zip_url: str,
        *,
        output_dir: Path | None = None,
        extract_dir: Path | None = None,
    ) -> MinerUParsedResult:
        zip_dir = output_dir or default_result_cache_dir(_result_id(full_zip_url))
        zip_dir.mkdir(parents=True, exist_ok=True)
        zip_path = zip_dir / "result.zip"
        with self._client.stream("GET", full_zip_url) as response:
            _ = response.raise_for_status()
            with zip_path.open("wb") as file:
                for chunk in response.iter_bytes():
                    _ = file.write(chunk)
        result_output_dir = extract_dir or output_dir or zip_dir
        return MinerUParsedResult.from_zip_file(
            zip_path,
            result_output_dir,
            extract_dir=extract_dir,
        )

    def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: Mapping[str, Json] | None = None,
    ) -> dict[str, object]:
        response = self._client.request(
            method,
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "*/*",
            },
            json=json_body,
        )
        _ = response.raise_for_status()
        body = cast(dict[str, object], response.json())
        if body.get("code") != 0:
            raise MinerUApiError(
                _required_int_or_str(body, "code"),
                _optional_str(body, "msg") or "",
                _optional_str(body, "trace_id"),
            )
        data = body["data"]
        if not isinstance(data, dict):
            raise TypeError("Expected data to be object")
        return cast(dict[str, object], data)


type TransportHandler = Callable[[httpx.Request], httpx.Response]


def _compact(data: Mapping[str, Json | None]) -> dict[str, Json]:
    return {key: value for key, value in data.items() if value is not None}


def _optional_str(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Expected {key} to be str")
    return value


def _required_int_or_str(data: Mapping[str, object], key: str) -> int | str:
    value = data[key]
    if not isinstance(value, int | str):
        raise TypeError(f"Expected {key} to be int or str")
    return value


def _batch_item_data_id(index: int) -> str:
    return f"uminer-{index + 1}"


def _result_id(full_zip_url: str) -> str:
    return hashlib.sha256(full_zip_url.encode("utf-8")).hexdigest()[:24]
