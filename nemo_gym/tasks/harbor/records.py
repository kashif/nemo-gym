# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two record files prepare writes next to the task folders, and how they are read.

compose-images.json records image configurations, built-images.json records the images built
per context digest and service. This module names the files and the service keys, takes the
sidecar lock, replaces a file atomically, reads it with one retry against a torn write, and
reads the per-service entries with the legacy main-only fallback. The loader, the recorders,
the build modules and the environment server all read through it.
"""

import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from nemo_gym.tasks.harbor.errors import HarborTaskError


# The OCI configurations recorded at prepare next to the task folders, keyed by image reference as the
# tasks wrote it (see ``nemo_gym.tasks.harbor.image_configs``): Compose sidecar images and the base image
# of every Dockerfile the loader resolves.
IMAGE_CONFIGS_FILE = "compose-images.json"
# The images prepare built for Dockerfiles outside pull and overlay mode, next to the task folders, keyed by
# Harbor's content hash of the task's ``environment/`` build context (see ``nemo_gym.tasks.harbor.image_build``).
BUILT_IMAGES_FILE = "built-images.json"

# The sidecar lock file next to a record file (``<file>.lock``), see :func:`record_lock`.
RECORD_LOCK_SUFFIX = ".lock"
# A reader that finds a record file unparsable reads it once more after this pause before giving up.
RECORD_READ_RETRY_S = 0.05


@contextmanager
def _flock(lock_path: Path, *, blocking: bool = True) -> Iterator[bool]:
    """Hold an exclusive ``flock`` on ``lock_path`` (created if missing), yielding whether it was taken.

    Blocking, the lock is always taken and ``True`` is yielded. Non-blocking, ``False`` is yielded at once
    when another process (or thread: ``flock`` is per open file description) holds it, and the caller does
    nothing under it.
    """
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def record_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``path``'s sidecar lock file (``<file>.lock``) for a read-modify-write of it.

    Two Gym processes can share one datasets folder (``gym eval run`` and ``gym dataset validate`` of one
    ``harbor:`` ref both record into it), so every read, merge and replace of a record file happens under
    this lock or a concurrent writer's entries are lost. ``flock`` is per open file description, so threads
    of one process are serialised as well. The lock file stays next to the record; it carries no content.
    """
    path = Path(path)
    with _flock(path.with_name(path.name + RECORD_LOCK_SUFFIX)):
        yield


def entry_lock_path(path: Path, key: str) -> Path:
    """The sidecar lock file for one entry of the record at ``path``: ``<file>.<sha256(key)[:16]>.lock``."""
    path = Path(path)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return path.with_name(f"{path.name}.{digest}{RECORD_LOCK_SUFFIX}")


@contextmanager
def entry_lock(path: Path, key: str, *, blocking: bool = True) -> Iterator[bool]:
    """Hold an exclusive lock on one entry of the record at ``path`` (:func:`entry_lock_path`), yielding whether
    it was taken (always ``True`` when ``blocking``).

    A recorder holds it from reading the record to writing the entry, around the slow work in between (a
    registry lookup), so a second process wanting the same entry waits for the first instead of doing that
    work again; it then reads the record and finds the entry there. :func:`record_lock` is taken inside it
    for the merge, never the other way round. The lock file is removed by :func:`discard_entry_lock` once the
    entry is recorded; until then it stays, so a process that opened it before the removal and one that opens
    it after lock different inodes only once the entry is on disk, when neither has anything left to do.
    """
    with _flock(entry_lock_path(path, key), blocking=blocking) as taken:
        yield taken


def discard_entry_lock(path: Path, key: str) -> None:
    """Remove the lock file of :func:`entry_lock`, by its holder, once the entry is recorded.

    Safe only while holding the lock and after the entry was seen on disk: every later holder reads the
    record first and finds it, whichever inode its lock file is.
    """
    try:
        os.unlink(entry_lock_path(path, key))
    except FileNotFoundError:
        pass


def write_json_record(path: Path, data: dict[str, Any]) -> None:
    """Replace the record file at ``path`` with ``data`` atomically: written sorted and indented to a temporary
    file in the same directory, fsynced, then moved over ``path`` with ``os.replace``, so a reader opens either
    the old file or the new one and never a truncated one. Callers merging into an existing record hold
    :func:`record_lock` around the read, the merge and this call."""
    path = Path(path)
    text = json.dumps(dict(sorted(data.items())), indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    # os.open with 0o666 applies the umask like Path.write_text did; mkstemp would make the record 0600.
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_json_record(path: Path) -> dict[str, Any]:
    """The JSON object in the record file at ``path``, or ``{}`` when the file is missing or not an object.

    Writers replace the file atomically (:func:`write_json_record`), so a torn read should not happen; as
    belt and braces a file that does not parse is read once more after a short pause (a record an older
    writer is still truncating and rewriting) and :class:`HarborTaskError` names it if it stays invalid.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    for attempt in (1, 2):
        try:
            recorded = json.loads(path.read_text())
            break
        except FileNotFoundError:
            return {}
        except (ValueError, UnicodeDecodeError) as exc:
            if attempt == 1:
                time.sleep(RECORD_READ_RETRY_S)
                continue
            raise HarborTaskError(
                f"{path} is not valid JSON (read twice, {RECORD_READ_RETRY_S:g}s apart): {exc}"
            ) from exc
    return recorded if isinstance(recorded, dict) else {}


def read_image_configs(folder: Path) -> dict[str, Any]:
    """The image configurations recorded in ``folder`` (the tasks' parent), or ``{}`` when none were."""
    return read_json_record(Path(folder) / IMAGE_CONFIGS_FILE)


def read_built_images(folder: Path) -> dict[str, Any]:
    """The built images recorded in ``folder`` (the tasks' parent), or ``{}`` when none were."""
    return read_json_record(Path(folder) / BUILT_IMAGES_FILE)
