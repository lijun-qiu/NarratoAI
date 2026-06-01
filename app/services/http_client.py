#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Shared HTTP session for ASR/transcription requests."""

from __future__ import annotations

from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import config


def get_request_proxies() -> Optional[dict[str, str]]:
    proxy_cfg = config.proxy if hasattr(config, "proxy") else {}
    if not proxy_cfg.get("enabled"):
        return None
    http_proxy = (proxy_cfg.get("http") or "").strip()
    https_proxy = (proxy_cfg.get("https") or http_proxy).strip()
    if not http_proxy and not https_proxy:
        return None
    return {
        "http": http_proxy or https_proxy,
        "https": https_proxy or http_proxy,
    }


def get_ssl_verify() -> bool:
    section = config.transcription if hasattr(config, "transcription") else {}
    if isinstance(section, dict) and "ssl_verify" in section:
        return bool(section.get("ssl_verify"))
    return True


def create_http_session(
    *,
    total_retries: int = 3,
    backoff_factor: float = 1.0,
) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=0,
        backoff_factor=backoff_factor,
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.proxies = get_request_proxies() or {}
    session.verify = get_ssl_verify()
    return session


def request_post(url: str, **kwargs: Any) -> requests.Response:
    session = kwargs.pop("session", None) or create_http_session()
    kwargs.setdefault("timeout", 300)
    if "verify" not in kwargs:
        kwargs["verify"] = get_ssl_verify()
    if "proxies" not in kwargs:
        proxies = get_request_proxies()
        if proxies:
            kwargs["proxies"] = proxies
    return session.post(url, **kwargs)


def request_get(url: str, **kwargs: Any) -> requests.Response:
    session = kwargs.pop("session", None) or create_http_session()
    kwargs.setdefault("timeout", 120)
    if "verify" not in kwargs:
        kwargs["verify"] = get_ssl_verify()
    if "proxies" not in kwargs:
        proxies = get_request_proxies()
        if proxies:
            kwargs["proxies"] = proxies
    return session.get(url, **kwargs)
