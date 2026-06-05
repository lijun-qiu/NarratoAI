"""Multi-instance path and port resolution for parallel WebUI processes."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass

DEFAULT_BASE_PORT = 8501
_STORAGE_SUBDIRS = ("temp", "tasks", "json", "narration_scripts", "drama_analysis")


@dataclass(frozen=True)
class InstancePaths:
    instance_id: str
    config_file: str
    storage_root: str
    port: int


def _sanitize_instance_id(raw: str) -> str:
    return re.sub(r"[^\w\-]", "", raw.strip())


def _resolve_port(instance_id: str) -> int:
    port_env = os.environ.get("NARRATO_PORT", "").strip()
    if port_env.isdigit():
        return int(port_env)
    if instance_id.isdigit():
        return DEFAULT_BASE_PORT + int(instance_id) - 1
    raise ValueError(
        f"实例 ID「{instance_id}」非数字，请通过环境变量 NARRATO_PORT 指定端口"
    )


def _ensure_instance_storage(storage_root: str) -> None:
    os.makedirs(storage_root, exist_ok=True)
    for sub_dir in _STORAGE_SUBDIRS:
        os.makedirs(os.path.join(storage_root, sub_dir), exist_ok=True)


def init_instance_paths(root_dir: str, base_config_file: str) -> InstancePaths:
    """Resolve config/storage/port for the current process."""
    raw_id = os.environ.get("NARRATO_INSTANCE_ID", "").strip()
    if not raw_id:
        port_env = os.environ.get("NARRATO_PORT", "").strip()
        port = int(port_env) if port_env.isdigit() else DEFAULT_BASE_PORT
        storage_root = os.path.join(root_dir, "storage")
        return InstancePaths(
            instance_id="",
            config_file=base_config_file,
            storage_root=storage_root,
            port=port,
        )

    instance_id = _sanitize_instance_id(raw_id)
    if not instance_id:
        raise ValueError("NARRATO_INSTANCE_ID 无效，仅允许字母、数字、下划线与连字符")

    instance_dir = os.path.join(root_dir, "instances", instance_id)
    os.makedirs(instance_dir, exist_ok=True)

    instance_config = os.path.join(instance_dir, "config.toml")
    if not os.path.isfile(instance_config) and os.path.isfile(base_config_file):
        shutil.copy2(base_config_file, instance_config)

    storage_root = os.path.join(instance_dir, "storage")
    _ensure_instance_storage(storage_root)

    return InstancePaths(
        instance_id=instance_id,
        config_file=instance_config,
        storage_root=storage_root,
        port=_resolve_port(instance_id),
    )
