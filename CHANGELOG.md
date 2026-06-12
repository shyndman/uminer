## v0.2.1 (2026-06-12)

### Fix

- **results**: extract result zips atomically
- **api**: extend upload write timeout to 120s

## v0.2.0 (2026-05-27)

### Feat

- **cli**: write local extracts beside source
- **cli**: support multi-source extract
- **cli**: download completed tasks by id
- **cli**: persist completed extract status steps
- **cli**: improve extract command presentation
- **api**: add upload progress tracking
- **cli**: print result zip url during extract

### Fix

- **release**: restore released version metadata
- **api**: type task ids as UUIDs
- **job**: retry transient 403 batch-result polls
- **cli**: display "fail" instead of "failed" in list table

### Refactor

- **tests**: migrate test_api from unittest to pytest
- **tests**: migrate test_api from unittest to pytest

## v0.1.0 (2026-05-23)

### Feat

- **cli**: show extract progress and styled tasks

## v0.0.2 (2026-05-21)

### Fix

- **release**: restore git write access after bump
- **release**: lock bumped versions before pushing
- **release**: lock bumped versions before pushing

## v0.0.1 (2026-05-21)

### Fix

- **release**: push bootstrap tag with the PAT

## v0.0.0 (2026-05-21)

### Feat

- **release**: automate versioning and publishing
- **project**: rename package to uminer
- add task listing endpoint
- store extraction results on disk
- add typed extraction models
- add extraction result jobs
- add MinerU API client

### Fix

- **release**: disable incremental bootstrap changelogs
- **release**: handle bootstrap bump previews
- **release**: install uv in bump job
- restrict commit-msg hooks
- satisfy ruff lint checks
