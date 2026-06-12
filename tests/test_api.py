from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import zipfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import cast
from uuid import UUID

import httpx
import pytest
from pydantic import ValidationError

from uminer import (
    ExtractionBatch,
    ExtractionBatchItemResult,
    ExtractionJob,
    MinerUApiError,
    MinerUClient,
    MinerUConfigError,
    MinerUParsedResult,
    MinerUTaskFailedError,
    ParagraphBlock,
)

FAKE_TASK_ID = "550e8400-e29b-41d4-a716-446655440000"
FAKE_TASK_ID_2 = "550e8400-e29b-41d4-a716-446655440001"


def _mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(
        base_url="https://mineru.net", transport=httpx.MockTransport(handler)
    )


def _result_zip_bytes() -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("full.md", "# Smoke\n")
        archive.writestr(
            "content_list_v2.json",
            json.dumps(
                [
                    [
                        {
                            "type": "paragraph",
                            "content": {
                                "paragraph_content": [
                                    {"type": "text", "content": "Smoke"}
                                ]
                            },
                        }
                    ]
                ]
            ),
        )
        archive.writestr(
            "demo_model.json", json.dumps([[{"type": "text", "content": "Smoke"}]])
        )
        archive.writestr("layout.json", json.dumps({"_backend": "vlm", "pdf_info": []}))
    return output.getvalue()


async def _await_job(job: ExtractionJob) -> MinerUParsedResult:
    return await job


def _json_response(data: Mapping[str, object]) -> httpx.Response:
    return httpx.Response(
        200, json={"code": 0, "msg": "ok", "trace_id": "trace-1", "data": data}
    )


def _ok_response(_request: httpx.Request) -> httpx.Response:
    return _json_response({"task_id": FAKE_TASK_ID})


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINERU_API_KEY", raising=False)
    with pytest.raises(MinerUConfigError):
        _ = MinerUClient()


def test_uses_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINERU_API_KEY", "env-token")
    client = MinerUClient(client=_mock_client(_ok_response))
    assert client.api_key == "env-token"


def test_create_extract_task_posts_expected_body() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/api/v4/extract/task"
        assert request.headers["authorization"] == "Bearer token"
        assert json.loads(request.content) == {
            "url": "https://example.com/demo.pdf",
            "model_version": "vlm",
            "enable_table": True,
            "extra_formats": ["docx", "html"],
        }
        return _json_response({"task_id": FAKE_TASK_ID})

    client = MinerUClient(api_key="token", client=_mock_client(handler))
    task = client.create_extract_task(
        "https://example.com/demo.pdf",
        enable_table=True,
        extra_formats=["docx", "html"],
    )

    assert task.task_id == UUID(FAKE_TASK_ID)
    assert len(requests) == 1


def test_get_extract_task_maps_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == f"/api/v4/extract/task/{FAKE_TASK_ID}"
        return _json_response(
            {
                "task_id": FAKE_TASK_ID,
                "state": "running",
                "err_msg": "",
                "extract_progress": {
                    "extracted_pages": 1,
                    "total_pages": 2,
                    "start_time": "2025-01-20 11:43:20",
                },
            }
        )

    client = MinerUClient(api_key="token", client=_mock_client(handler))
    task = client.get_extract_task(FAKE_TASK_ID)

    assert task.state == "running"
    assert task.extract_progress is not None
    assert task.extract_progress.extracted_pages == 1


def test_list_tasks_maps_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v4/tasks"
        assert dict(request.url.params) == {"page_no": "2", "page_size": "5"}
        return _json_response(
            {
                "list": [
                    {
                        "file_name": "demo.pdf",
                        "task_id": FAKE_TASK_ID,
                        "type": "pdf",
                        "state": "done",
                        "full_md_link": "https://cdn.example/full.md",
                        "err_msg": "",
                        "created_at": 1778173950469,
                        "model_version": "vlm2.7.6",
                        "file_size": 2046627,
                        "is_chem": False,
                        "can_retry": False,
                        "rank": 0,
                        "is_expire": False,
                        "cover_path": "https://cdn.example/cover.webp",
                        "file_url": "",
                    },
                    {
                        "file_name": "broken.pdf",
                        "task_id": FAKE_TASK_ID_2,
                        "type": "",
                        "state": "failed",
                        "full_md_link": "",
                        "err_msg": "parsing failed, please try again later",
                        "err_code": -60010,
                        "created_at": 1779025960944,
                        "model_version": "vlm3.1.8",
                        "file_size": 1048576,
                        "is_chem": True,
                        "can_retry": True,
                        "rank": 0,
                        "is_expire": False,
                        "cover_path": "",
                        "file_url": "https://example.com/broken.pdf",
                    },
                ],
                "total": 2,
            }
        )

    client = MinerUClient(api_key="token", client=_mock_client(handler))
    page = client.list_tasks(page_no=2, page_size=5)

    assert page.total == 2
    assert len(page.tasks) == 2
    assert page.tasks[0].task_id == UUID(FAKE_TASK_ID)
    assert page.tasks[0].file_name == "demo.pdf"
    assert page.tasks[0].created_at == datetime.fromtimestamp(1778173950469 / 1000, UTC)
    assert str(page.tasks[0].full_md_link) == "https://cdn.example/full.md"
    assert page.tasks[0].file_url is None
    assert not page.tasks[0].has_chemical_formula
    assert page.tasks[0].error is None
    assert "rank" not in page.tasks[0].model_dump()

    assert page.tasks[1].task_id == UUID(FAKE_TASK_ID_2)
    assert page.tasks[1].file_type is None
    assert page.tasks[1].state == "failed"
    assert page.tasks[1].has_chemical_formula
    assert str(page.tasks[1].file_url) == "https://example.com/broken.pdf"
    assert page.tasks[1].error is not None
    assert page.tasks[1].error.code == -60010
    assert page.tasks[1].error.message == "parsing failed, please try again later"


def test_list_tasks_rejects_non_positive_paging() -> None:
    client = MinerUClient(api_key="token", client=_mock_client(_ok_response))

    with pytest.raises(ValidationError):
        _ = client.list_tasks(page_no=0)

    with pytest.raises(ValidationError):
        _ = client.list_tasks(page_size=0)


def test_create_upload_batch_and_upload_files(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    progress_updates: list[tuple[str, int, int]] = []

    def on_upload_progress(current_path: Path, uploaded: int, total: int) -> None:
        progress_updates.append((current_path.name, uploaded, total))

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            assert request.url.path == "/api/v4/file-urls/batch"
            assert json.loads(request.content) == {
                "files": [{"name": "demo.pdf", "data_id": "doc-1"}],
                "model_version": "vlm",
            }
            return _json_response(
                {
                    "batch_id": "batch-1",
                    "file_urls": ["https://uploads.example/demo.pdf"],
                }
            )
        assert request.method == "PUT"
        assert str(request.url) == "https://uploads.example/demo.pdf"
        assert request.headers["Content-Length"] == "9"
        assert request.content == b"pdf bytes"
        return httpx.Response(200)

    path = tmp_path / "demo.pdf"
    _ = path.write_bytes(b"pdf bytes")
    client = MinerUClient(api_key="token", client=_mock_client(handler))
    batch = client.create_file_upload_extract_tasks(
        [path],
        files=[{"name": "demo.pdf", "data_id": "doc-1"}],
        on_upload_progress=on_upload_progress,
    )

    assert batch.batch_id == "batch-1"
    assert batch.file_urls == ("https://uploads.example/demo.pdf",)
    assert [request.method for request in requests] == ["POST", "PUT"]
    assert progress_updates == [("demo.pdf", 9, 9)]


def test_upload_file_extends_write_timeout(tmp_path: Path) -> None:
    seen: list[dict[str, float | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(cast("dict[str, float | None]", request.extensions["timeout"]))
        return httpx.Response(200)

    path = tmp_path / "demo.pdf"
    _ = path.write_bytes(b"pdf bytes")
    client = MinerUClient(api_key="token", client=_mock_client(handler))
    client.upload_file("https://uploads.example/demo.pdf", path)

    assert len(seen) == 1
    timeout = seen[0]
    assert timeout["write"] == 120.0
    assert timeout["connect"] == 5.0
    assert timeout["read"] == 5.0


def test_create_url_batch_returns_batch_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v4/extract/task/batch"
        assert json.loads(request.content) == {
            "files": [{"url": "https://example.com/demo.pdf", "data_id": "doc-1"}],
            "model_version": "vlm",
            "no_cache": True,
        }
        return _json_response({"batch_id": "batch-1"})

    client = MinerUClient(api_key="token", client=_mock_client(handler))
    batch_id = client.create_url_batch(
        [{"url": "https://example.com/demo.pdf", "data_id": "doc-1"}],
        no_cache=True,
    )

    assert batch_id == "batch-1"


def test_get_batch_extract_result_maps_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v4/extract-results/batch/batch-1"
        return _json_response(
            {
                "batch_id": "batch-1",
                "extract_result": [
                    {
                        "file_name": "demo.pdf",
                        "state": "done",
                        "full_zip_url": "https://cdn.example/demo.zip",
                        "err_msg": "",
                    }
                ],
            }
        )

    client = MinerUClient(api_key="token", client=_mock_client(handler))
    result = client.get_batch_extract_result("batch-1")

    assert result.batch_id == "batch-1"
    assert result.results[0].file_name == "demo.pdf"
    assert result.results[0].full_zip_url == "https://cdn.example/demo.zip"


def test_api_error_raises_with_code_and_trace_id() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"code": "A0202", "msg": "Invalid Token", "trace_id": "trace-1"},
        )

    client = MinerUClient(api_key="token", client=_mock_client(handler))

    with pytest.raises(MinerUApiError) as exc_info:
        _ = client.get_extract_task(FAKE_TASK_ID)

    assert exc_info.value.code == "A0202"
    assert exc_info.value.trace_id == "trace-1"


def test_download_result_parses_zip_outputs(tmp_path: Path) -> None:
    zip_bytes = _result_zip_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://cdn.example/result.zip"
        return httpx.Response(200, content=zip_bytes)

    output_dir = tmp_path / "result"
    client = MinerUClient(api_key="token", client=_mock_client(handler))
    result = client.download_result(
        "https://cdn.example/result.zip", output_dir=output_dir
    )

    assert result.output_dir == output_dir
    assert result.zip_path == output_dir / "result.zip"
    assert result.markdown == "# Smoke\n"
    assert len(result.content_list.pages) == 1
    block = result.content_list.pages[0].blocks[0]
    assert isinstance(block, ParagraphBlock)
    assert block.content.paragraph_content[0].content == "Smoke"
    assert result.raw_output == [[{"type": "text", "content": "Smoke"}]]
    assert result.layout == {"_backend": "vlm", "pdf_info": []}
    full_md = next(file for file in result.files if file.path == "full.md")
    assert full_md.local_path.exists()


def test_download_result_swaps_atomically_and_clears_stale_output(
    tmp_path: Path,
) -> None:
    zip_bytes = _result_zip_bytes()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=zip_bytes)

    output_dir = tmp_path / "result"
    extract_dir = output_dir
    extract_dir.mkdir(parents=True)
    stale = extract_dir / "stale.md"
    _ = stale.write_text("old run", encoding="utf-8")
    leftover_staging = output_dir.with_name(output_dir.name + ".tmp")
    leftover_staging.mkdir()
    _ = (leftover_staging / "junk.txt").write_text("abandoned", encoding="utf-8")

    client = MinerUClient(api_key="token", client=_mock_client(handler))
    result = client.download_result(
        "https://cdn.example/result.zip",
        output_dir=output_dir,
        extract_dir=extract_dir,
    )

    assert result.output_dir == output_dir
    assert not stale.exists()
    assert (extract_dir / "full.md").read_text(encoding="utf-8") == "# Smoke\n"
    assert not leftover_staging.exists()


def test_extract_url_job_reports_status_and_waits_for_result() -> None:
    requests: list[str] = []
    zip_bytes = _result_zip_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(f"{request.method} {request.url}")
        if request.method == "POST":
            return _json_response({"task_id": FAKE_TASK_ID})
        if str(request.url).endswith(f"/api/v4/extract/task/{FAKE_TASK_ID}"):
            return _json_response(
                {
                    "task_id": FAKE_TASK_ID,
                    "state": "done",
                    "full_zip_url": "https://cdn.example/result.zip",
                    "err_msg": "",
                }
            )
        return httpx.Response(200, content=zip_bytes)

    client = MinerUClient(api_key="token", client=_mock_client(handler))
    job = client.extract_url("https://example.com/demo.pdf", poll_interval_seconds=0)

    assert job.source.kind == "url"
    assert job.source.url == "https://example.com/demo.pdf"
    assert job.last_status.task_id == UUID(FAKE_TASK_ID)
    assert job.last_status.state is None
    status = job()
    result = job.wait()
    awaited = asyncio.run(_await_job(job))

    assert status.state == "done"
    assert job.last_status.state == "done"
    assert job.last_status.full_zip_url == "https://cdn.example/result.zip"
    assert result.markdown == "# Smoke\n"
    assert awaited.markdown == "# Smoke\n"
    assert requests[0] == "POST https://mineru.net/api/v4/extract/task"


def test_extract_file_job_reports_batch_status(tmp_path: Path) -> None:
    zip_bytes = _result_zip_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return _json_response(
                {
                    "batch_id": "batch-1",
                    "file_urls": ["https://uploads.example/demo.pdf"],
                }
            )
        if request.method == "PUT":
            return httpx.Response(200)
        if str(request.url).endswith("/api/v4/extract-results/batch/batch-1"):
            return _json_response(
                {
                    "batch_id": "batch-1",
                    "extract_result": [
                        {
                            "file_name": "demo.pdf",
                            "state": "done",
                            "full_zip_url": "https://cdn.example/result.zip",
                            "err_msg": "",
                        }
                    ],
                }
            )
        return httpx.Response(200, content=zip_bytes)

    path = tmp_path / "demo.pdf"
    _ = path.write_bytes(b"pdf bytes")
    client = MinerUClient(api_key="token", client=_mock_client(handler))
    job = client.extract_file(path, poll_interval_seconds=0)

    assert job.source.kind == "file"
    assert job.source.path == path
    assert job.source.file == {"name": "demo.pdf"}
    assert job.last_status.batch_id == "batch-1"
    assert job.last_status.file_name == "demo.pdf"
    assert job().state == "done"
    assert job.last_status.state == "done"
    assert job.wait().markdown == "# Smoke\n"


def test_extract_file_job_writes_default_output_beside_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    zip_url = "https://cdn.example/result.zip"
    zip_bytes = _result_zip_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return _json_response(
                {
                    "batch_id": "batch-1",
                    "file_urls": ["https://uploads.example/demo.pdf"],
                }
            )
        if request.method == "PUT":
            return httpx.Response(200)
        if str(request.url).endswith("/api/v4/extract-results/batch/batch-1"):
            return _json_response(
                {
                    "batch_id": "batch-1",
                    "extract_result": [
                        {
                            "file_name": "demo.pdf",
                            "state": "done",
                            "full_zip_url": zip_url,
                            "err_msg": "",
                        }
                    ],
                }
            )
        assert str(request.url) == zip_url
        return httpx.Response(200, content=zip_bytes)

    path = tmp_path / "demo.pdf"
    _ = path.write_bytes(b"pdf bytes")
    client = MinerUClient(api_key="token", client=_mock_client(handler))

    result = client.extract_file(path, poll_interval_seconds=0).wait()

    result_id = hashlib.sha256(zip_url.encode("utf-8")).hexdigest()[:24]
    assert result.output_dir == tmp_path / "demo.pdf.uminer"
    assert result.zip_path == (
        tmp_path / "cache" / "uminer" / "results" / result_id / "result.zip"
    )
    assert (result.output_dir / "full.md").read_text(encoding="utf-8") == "# Smoke\n"


def test_extract_files_write_default_outputs_beside_each_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    zip_bytes = _result_zip_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return _json_response(
                {
                    "batch_id": "batch-1",
                    "file_urls": [
                        "https://uploads.example/one.pdf",
                        "https://uploads.example/two.pdf",
                    ],
                }
            )
        if request.method == "PUT":
            return httpx.Response(200)
        if str(request.url).endswith("/api/v4/extract-results/batch/batch-1"):
            return _json_response(
                {
                    "batch_id": "batch-1",
                    "extract_result": [
                        {
                            "file_name": "one.pdf",
                            "data_id": "uminer-1",
                            "state": "done",
                            "full_zip_url": "https://cdn.example/one.zip",
                            "err_msg": "",
                        },
                        {
                            "file_name": "two.pdf",
                            "data_id": "uminer-2",
                            "state": "done",
                            "full_zip_url": "https://cdn.example/two.zip",
                            "err_msg": "",
                        },
                    ],
                }
            )
        assert str(request.url) in {
            "https://cdn.example/one.zip",
            "https://cdn.example/two.zip",
        }
        return httpx.Response(200, content=zip_bytes)

    first = tmp_path / "one.pdf"
    second = tmp_path / "two.pdf"
    _ = first.write_bytes(b"one")
    _ = second.write_bytes(b"two")
    client = MinerUClient(api_key="token", client=_mock_client(handler))

    results = client.extract_files([first, second], poll_interval_seconds=0).wait()

    assert [result.output_dir for result in results] == [
        tmp_path / "one.pdf.uminer",
        tmp_path / "two.pdf.uminer",
    ]
    assert all(result.result is not None for result in results)
    assert all(
        result.result is not None and result.result.zip_path.parent != result.output_dir
        for result in results
    )


def test_extract_urls_waits_for_each_batch_item_and_collects_failures(
    tmp_path: Path,
) -> None:
    zip_bytes = _result_zip_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            assert request.url.path == "/api/v4/extract/task/batch"
            return _json_response({"batch_id": "batch-1"})
        if str(request.url).endswith("/api/v4/extract-results/batch/batch-1"):
            return _json_response(
                {
                    "batch_id": "batch-1",
                    "extract_result": [
                        {
                            "file_name": "demo.pdf",
                            "data_id": "uminer-1",
                            "state": "done",
                            "full_zip_url": "https://cdn.example/demo.zip",
                            "err_msg": "",
                        },
                        {
                            "file_name": "broken.pdf",
                            "data_id": "uminer-2",
                            "state": "failed",
                            "err_msg": "Unsupported file",
                        },
                    ],
                }
            )
        assert str(request.url) == "https://cdn.example/demo.zip"
        return httpx.Response(200, content=zip_bytes)

    client = MinerUClient(api_key="token", client=_mock_client(handler))
    batch = client.extract_urls(
        ["https://example.com/demo.pdf", "https://example.com/broken.pdf"],
        poll_interval_seconds=0,
    )

    assert isinstance(batch, ExtractionBatch)
    results = batch.wait(output_dir=tmp_path / "out")

    assert len(results) == 2
    assert results[0].ok
    assert results[0].output_dir == tmp_path / "out" / "001-demo.pdf"
    assert results[0].result is not None
    assert results[0].result.markdown == "# Smoke\n"
    assert not results[1].ok
    assert isinstance(results[1].error, MinerUTaskFailedError)
    assert results[1].status.state == "failed"
    assert results[1].status.err_msg == "Unsupported file"


def test_extract_files_uploads_in_parallel(tmp_path: Path) -> None:
    first_started = threading.Event()
    second_started = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return _json_response(
                {
                    "batch_id": "batch-1",
                    "file_urls": [
                        "https://uploads.example/one.pdf",
                        "https://uploads.example/two.pdf",
                    ],
                }
            )
        if str(request.url) == "https://uploads.example/one.pdf":
            first_started.set()
            assert second_started.wait(0.5)
            return httpx.Response(200)
        assert str(request.url) == "https://uploads.example/two.pdf"
        second_started.set()
        return httpx.Response(200)

    first = tmp_path / "one.pdf"
    second = tmp_path / "two.pdf"
    _ = first.write_bytes(b"one")
    _ = second.write_bytes(b"two")
    client = MinerUClient(api_key="token", client=_mock_client(handler))

    batch = client.extract_files([first, second], max_upload_workers=2)

    assert isinstance(batch, ExtractionBatch)
    assert batch.item_results == (None, None)
    assert first_started.is_set()
    assert second_started.is_set()


def test_extract_files_preserves_upload_failures_and_waits_for_successes(
    tmp_path: Path,
) -> None:
    zip_bytes = _result_zip_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return _json_response(
                {
                    "batch_id": "batch-1",
                    "file_urls": [
                        "https://uploads.example/one.pdf",
                        "https://uploads.example/two.pdf",
                    ],
                }
            )
        if str(request.url) == "https://uploads.example/one.pdf":
            return httpx.Response(500, request=request, text="boom")
        if str(request.url).endswith("/api/v4/extract-results/batch/batch-1"):
            return _json_response(
                {
                    "batch_id": "batch-1",
                    "extract_result": [
                        {
                            "file_name": "two.pdf",
                            "data_id": "uminer-2",
                            "state": "done",
                            "full_zip_url": "https://cdn.example/two.zip",
                            "err_msg": "",
                        }
                    ],
                }
            )
        if str(request.url) == "https://cdn.example/two.zip":
            return httpx.Response(200, content=zip_bytes)
        assert str(request.url) == "https://uploads.example/two.pdf"
        return httpx.Response(200)

    first = tmp_path / "one.pdf"
    second = tmp_path / "two.pdf"
    _ = first.write_bytes(b"one")
    _ = second.write_bytes(b"two")
    client = MinerUClient(api_key="token", client=_mock_client(handler))

    batch = client.extract_files([first, second], poll_interval_seconds=0)
    results = batch.wait(output_dir=tmp_path / "out")

    assert len(results) == 2
    assert not results[0].ok
    assert results[0].status.state == "failed"
    assert results[0].result is None
    assert results[0].output_dir is None
    assert results[0].error is not None
    assert isinstance(results[0], ExtractionBatchItemResult)
    assert results[1].ok
    assert results[1].output_dir == tmp_path / "out" / "002-two.pdf"
    assert results[1].result is not None
    assert results[1].result.markdown == "# Smoke\n"


def test_extract_file_job_retries_transient_batch_403(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    batch_result_calls = 0
    sleep_calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal batch_result_calls
        if request.method == "POST":
            return _json_response(
                {
                    "batch_id": "batch-1",
                    "file_urls": ["https://uploads.example/demo.pdf"],
                }
            )
        if request.method == "PUT":
            return httpx.Response(200, request=request)
        if str(request.url).endswith("/api/v4/extract-results/batch/batch-1"):
            batch_result_calls += 1
            if batch_result_calls <= 3:
                return httpx.Response(403, request=request, text="forbidden")
            return _json_response(
                {
                    "batch_id": "batch-1",
                    "extract_result": [
                        {
                            "file_name": "demo.pdf",
                            "state": "done",
                            "full_zip_url": "https://cdn.example/demo.zip",
                            "err_msg": "",
                        }
                    ],
                }
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    monkeypatch.setattr("uminer.job.time.sleep", sleep_calls.append)

    path = tmp_path / "demo.pdf"
    _ = path.write_bytes(b"pdf bytes")
    client = MinerUClient(api_key="token", client=_mock_client(handler))
    job = client.extract_file(path, poll_interval_seconds=0)

    status = job()

    assert status.state == "done"
    assert batch_result_calls == 4
    assert sleep_calls == [10.0, 10.0, 10.0]


def test_extract_file_job_raises_after_six_transient_batch_403_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    batch_result_calls = 0
    sleep_calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal batch_result_calls
        if request.method == "POST":
            return _json_response(
                {
                    "batch_id": "batch-1",
                    "file_urls": ["https://uploads.example/demo.pdf"],
                }
            )
        if request.method == "PUT":
            return httpx.Response(200, request=request)
        if str(request.url).endswith("/api/v4/extract-results/batch/batch-1"):
            batch_result_calls += 1
            return httpx.Response(403, request=request, text="forbidden")
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    monkeypatch.setattr("uminer.job.time.sleep", sleep_calls.append)

    path = tmp_path / "demo.pdf"
    _ = path.write_bytes(b"pdf bytes")
    client = MinerUClient(api_key="token", client=_mock_client(handler))
    job = client.extract_file(path, poll_interval_seconds=0)

    with pytest.raises(httpx.HTTPStatusError):
        _ = job()

    assert batch_result_calls == 7
    assert sleep_calls == [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]


def test_wait_raises_for_failed_task() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return _json_response({"task_id": FAKE_TASK_ID})
        return _json_response(
            {"task_id": FAKE_TASK_ID, "state": "failed", "err_msg": "Unsupported file"}
        )

    client = MinerUClient(api_key="token", client=_mock_client(handler))
    job = client.extract_url("https://example.com/demo.pdf", poll_interval_seconds=0)

    with pytest.raises(MinerUTaskFailedError) as exc_info:
        _ = job.wait()

    assert exc_info.value.task_id == FAKE_TASK_ID
