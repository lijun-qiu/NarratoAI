"""
OpenAI 兼容提供商实现

使用 OpenAI 官方 SDK 调用 OpenAI 兼容接口，支持文本和视觉模型。
"""

import asyncio
import io
import base64
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import PIL.Image
from loguru import logger
from openai import (
    APIError as OpenAIAPIError,
    AsyncOpenAI,
    AuthenticationError as OpenAIAuthError,
    BadRequestError as OpenAIBadRequestError,
    RateLimitError as OpenAIRateLimitError,
)

from app.config import config
from app.config.defaults import normalize_openai_compatible_model_name
from .base import TextModelProvider, VisionModelProvider
from .exceptions import APICallError, AuthenticationError, ContentFilterError, RateLimitError


def _normalize_model_name(model_name: str) -> str:
    """仅剥离误保存的 openai/ 前缀，保留完整模型名称。"""
    return normalize_openai_compatible_model_name(model_name)


def _is_response_format_error(message: str) -> bool:
    return "response_format" in (message or "").lower()


def _is_content_filter_error(message: str) -> bool:
    lowered = (message or "").lower()
    return "content_filter" in lowered or "safety" in lowered


def _is_timeout_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "timed out" in msg or "timeout" in msg


def _is_retryable_transport_error(exc: Exception) -> bool:
    if _is_timeout_error(exc):
        return True
    msg = str(exc).lower()
    exc_name = type(exc).__name__.lower()
    retry_markers = (
        "connection error",
        "readerror",
        "connecterror",
        "remote protocol error",
        "server disconnected",
        "connection reset",
        "broken pipe",
        "unexpected eof",
    )
    return any(marker in msg or marker in exc_name for marker in retry_markers)


def resolve_llm_timeout(
    *,
    for_script: bool = False,
    timeout_override: Optional[float] = None,
) -> float:
    """解析 LLM 请求超时（秒）。脚本生成 prompt 较长，默认单独加长。"""
    if timeout_override is not None:
        return float(timeout_override)
    if for_script:
        script_timeout = config.app.get("llm_script_timeout")
        if script_timeout is not None:
            return float(script_timeout)
        text_timeout = float(config.app.get("llm_text_timeout", 180))
        return max(600.0, text_timeout * 2)
    return float(config.app.get("llm_text_timeout", 180))


def _clean_json_output(output: str) -> str:
    """清理 JSON 输出中的 markdown 包裹。"""
    output = re.sub(r"^```json\s*", "", output, flags=re.MULTILINE)
    output = re.sub(r"^```\s*$", "", output, flags=re.MULTILINE)
    output = re.sub(r"^```.*$", "", output, flags=re.MULTILINE)
    return output.strip()


class _OpenAICompatibleBase:
    """OpenAI 兼容 provider 共享逻辑。"""

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def supported_models(self) -> List[str]:
        # 兼容网关模型数量很多，运行时校验由远端完成。
        return []

    def _validate_model_support(self):
        logger.debug(f"OpenAI 兼容模型已配置: {self.model_name}")

    def _initialize(self):
        # SDK client 按请求参数动态构建，这里无需初始化全局状态。
        pass

    def _build_client(
        self,
        api_key_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        timeout_override: Optional[float] = None,
    ) -> AsyncOpenAI:
        """按请求构建 AsyncOpenAI 客户端，支持动态覆盖 api_key / base_url。"""
        api_key = api_key_override or self.api_key
        base_url = base_url_override or self.base_url or None

        timeout_seconds: float = resolve_llm_timeout(
            timeout_override=timeout_override,
        )
        max_retries: int = config.app.get("llm_max_retries", 3)

        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @asynccontextmanager
    async def _open_client(
        self,
        api_key_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        timeout_override: Optional[float] = None,
    ) -> AsyncIterator[AsyncOpenAI]:
        """构建并在使用后关闭 AsyncOpenAI，避免事件循环关闭后 httpx 清理报错。"""
        client = self._build_client(
            api_key_override=api_key_override,
            base_url_override=base_url_override,
            timeout_override=timeout_override,
        )
        try:
            yield client
        finally:
            await client.close()


class OpenAICompatibleVisionProvider(_OpenAICompatibleBase, VisionModelProvider):
    """OpenAI 兼容视觉模型提供商。"""

    async def analyze_images(
        self,
        images: List[Union[str, Path, PIL.Image.Image]],
        prompt: str,
        batch_size: int = 10,
        max_concurrency: int = 1,
        **kwargs,
    ) -> List[str]:
        logger.info(f"开始使用 OpenAI 兼容接口 ({self.model_name}) 分析 {len(images)} 张图片")

        processed_images = self._prepare_images(images)
        if not processed_images:
            return []

        bounded_concurrency = max(1, int(max_concurrency))
        semaphore = asyncio.Semaphore(bounded_concurrency)
        batches = [
            (index // batch_size, processed_images[index : index + batch_size])
            for index in range(0, len(processed_images), batch_size)
        ]

        async def run_batch(batch_index: int, batch: List[PIL.Image.Image]) -> tuple[int, str]:
            logger.info(f"处理第 {batch_index + 1} 批，共 {len(batch)} 张图片")
            async with semaphore:
                try:
                    result = await self._analyze_batch(batch, prompt, **kwargs)
                    return batch_index, result
                except Exception as exc:
                    logger.error(f"批次 {batch_index + 1} 处理失败: {exc}")
                    return batch_index, f"批次处理失败: {exc}"

        completed = await asyncio.gather(*(run_batch(index, batch) for index, batch in batches))
        completed.sort(key=lambda item: item[0])
        return [result for _, result in completed]

    async def _analyze_batch(self, batch: List[PIL.Image.Image], prompt: str, **kwargs) -> str:
        content = [{"type": "text", "text": prompt}]
        for img in batch:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self._image_to_base64(img)}"},
                }
            )

        messages = [{"role": "user", "content": content}]
        model_name = _normalize_model_name(self.model_name)

        async with self._open_client(
            api_key_override=kwargs.get("api_key"),
            base_url_override=kwargs.get("api_base"),
            timeout_override=config.app.get("llm_vision_timeout", 120),
        ) as client:
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=kwargs.get("temperature", 1.0),
                    max_tokens=kwargs.get("max_tokens", 4000),
                )
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    return response.choices[0].message.content
                raise APICallError("OpenAI 兼容接口返回空响应")
            except OpenAIAuthError as exc:
                logger.error(f"OpenAI 兼容接口认证失败: {exc}")
                raise AuthenticationError(str(exc))
            except OpenAIRateLimitError as exc:
                logger.error(f"OpenAI 兼容接口速率限制: {exc}")
                raise RateLimitError(str(exc))
            except OpenAIBadRequestError as exc:
                error_msg = str(exc)
                if _is_content_filter_error(error_msg):
                    raise ContentFilterError(f"内容被安全过滤器阻止: {error_msg}")
                raise APICallError(f"请求错误: {error_msg}")
            except OpenAIAPIError as exc:
                logger.error(f"OpenAI 兼容接口 API 错误: {exc}")
                raise APICallError(f"API 错误: {exc}")
            except Exception as exc:
                logger.error(f"OpenAI 兼容接口调用失败: {exc}")
                raise APICallError(f"调用失败: {exc}")

    def _image_to_base64(self, img: PIL.Image.Image) -> str:
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG", quality=85)
        return base64.b64encode(img_buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _video_to_data_url(video_path: Union[str, Path]) -> str:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"视频文件不存在: {path}")
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:video/mp4;base64,{encoded}"

    @staticmethod
    def _reference_image_to_data_url(image_path: Union[str, Path]) -> str:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"参照图不存在: {path}")
        suffix = path.suffix.lower()
        if suffix in (".png",):
            mime = "image/png"
        elif suffix in (".webp",):
            mime = "image/webp"
        else:
            mime = "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    async def analyze_video(
        self,
        video_path: Union[str, Path],
        prompt: str,
        **kwargs,
    ) -> str:
        """直接分析 mp4 视频（网关需支持 data:video/mp4 base64）。"""
        path = Path(video_path)
        size_mb = path.stat().st_size / (1024 * 1024)
        reference_paths = [
            str(item).strip()
            for item in (kwargs.get("reference_image_paths") or [])
            if str(item).strip()
        ]
        logger.info(
            f"视频分析上传: {path.name} ({size_mb:.2f} MB)"
            + (f"，参照图 {len(reference_paths)} 张" if reference_paths else "")
        )

        data_url = self._video_to_data_url(path)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for ref_path in reference_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._reference_image_to_data_url(ref_path)},
                }
            )
        content.append({"type": "image_url", "image_url": {"url": data_url}})
        messages = [{"role": "user", "content": content}]
        model_name = _normalize_model_name(self.model_name)
        base_timeout = float(
            kwargs.get("timeout_override") or config.app.get("llm_vision_timeout", 120)
        )
        timeout_retries = max(3, int(config.app.get("llm_timeout_retries", 2)))
        last_error: Optional[Exception] = None

        for attempt in range(timeout_retries):
            timeout_seconds = base_timeout * (1.0 + 0.5 * attempt)
            if attempt > 0:
                wait_seconds = min(30.0, 2.0 ** attempt)
                logger.warning(
                    f"视频分析连接异常，{wait_seconds:.0f}s 后重试 "
                    f"({attempt + 1}/{timeout_retries})，超时 {timeout_seconds:.0f}s"
                )
                await asyncio.sleep(wait_seconds)

            async with self._open_client(
                api_key_override=kwargs.get("api_key"),
                base_url_override=kwargs.get("api_base"),
                timeout_override=timeout_seconds,
            ) as client:
                try:
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=kwargs.get("temperature", 0.2),
                        max_tokens=kwargs.get("max_tokens", 8000),
                    )
                    if response.choices and response.choices[0].message and response.choices[0].message.content:
                        return response.choices[0].message.content
                    raise APICallError("OpenAI 兼容接口返回空响应")
                except OpenAIAuthError as exc:
                    raise AuthenticationError(str(exc))
                except OpenAIRateLimitError as exc:
                    raise RateLimitError(str(exc))
                except OpenAIBadRequestError as exc:
                    error_msg = str(exc)
                    if _is_content_filter_error(error_msg):
                        raise ContentFilterError(f"内容被安全过滤器阻止: {error_msg}")
                    raise APICallError(f"请求错误: {error_msg}")
                except OpenAIAPIError as exc:
                    last_error = exc
                    if _is_retryable_transport_error(exc) and attempt < timeout_retries - 1:
                        continue
                    raise APICallError(f"API 错误: {exc}")
                except Exception as exc:
                    last_error = exc
                    if _is_retryable_transport_error(exc) and attempt < timeout_retries - 1:
                        continue
                    raise APICallError(f"调用失败: {exc}")

        if last_error:
            raise APICallError(f"API 错误: {last_error}")
        raise APICallError("视频分析调用失败")

    async def _make_api_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload


class OpenAICompatibleTextProvider(_OpenAICompatibleBase, TextModelProvider):
    """OpenAI 兼容文本模型提供商。"""

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **kwargs,
    ) -> str:
        messages = self._build_messages(prompt, system_prompt)
        model_name = _normalize_model_name(kwargs.get("model") or self.model_name)

        completion_kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            completion_kwargs["max_tokens"] = max_tokens
        if response_format == "json":
            completion_kwargs["response_format"] = {"type": "json_object"}

        for_script = bool(kwargs.get("for_script", False))
        base_timeout = resolve_llm_timeout(
            for_script=for_script,
            timeout_override=kwargs.get("timeout_override"),
        )
        timeout_retries = max(1, int(config.app.get("llm_timeout_retries", 2)))
        last_error: Optional[Exception] = None

        for attempt in range(timeout_retries):
            timeout_seconds = base_timeout * (1.0 + 0.5 * attempt)
            if attempt > 0:
                logger.warning(
                    f"LLM 请求超时，使用 {timeout_seconds:.0f}s 超时重试 "
                    f"({attempt + 1}/{timeout_retries})"
                )

            async with self._open_client(
                api_key_override=kwargs.get("api_key"),
                base_url_override=kwargs.get("api_base"),
                timeout_override=timeout_seconds,
            ) as client:
                try:
                    response = await client.chat.completions.create(**completion_kwargs)
                    if response.choices and response.choices[0].message and response.choices[0].message.content:
                        return response.choices[0].message.content
                    raise APICallError("OpenAI 兼容接口返回空响应")

                except OpenAIBadRequestError as exc:
                    error_msg = str(exc)
                    if response_format == "json" and _is_response_format_error(error_msg):
                        logger.warning("目标网关不支持 response_format，回退为提示词约束 JSON 输出")
                        completion_kwargs.pop("response_format", None)
                        messages[-1]["content"] += "\n\n请确保输出严格的JSON格式，不要包含任何其他文字或标记。"

                        retry_response = await client.chat.completions.create(**completion_kwargs)
                        if retry_response.choices and retry_response.choices[0].message and retry_response.choices[0].message.content:
                            return _clean_json_output(retry_response.choices[0].message.content)
                        raise APICallError("OpenAI 兼容接口返回空响应")

                    if _is_content_filter_error(error_msg):
                        raise ContentFilterError(f"内容被安全过滤器阻止: {error_msg}")
                    raise APICallError(f"请求错误: {error_msg}")

                except OpenAIAuthError as exc:
                    logger.error(f"OpenAI 兼容接口认证失败: {exc}")
                    raise AuthenticationError(str(exc))
                except OpenAIRateLimitError as exc:
                    logger.error(f"OpenAI 兼容接口速率限制: {exc}")
                    raise RateLimitError(str(exc))
                except OpenAIAPIError as exc:
                    last_error = exc
                    if _is_timeout_error(exc) and attempt < timeout_retries - 1:
                        continue
                    logger.error(f"OpenAI 兼容接口 API 错误: {exc}")
                    raise APICallError(f"API 错误: {exc}")
                except Exception as exc:
                    last_error = exc
                    if _is_timeout_error(exc) and attempt < timeout_retries - 1:
                        continue
                    logger.error(f"OpenAI 兼容接口调用失败: {exc}")
                    raise APICallError(f"调用失败: {exc}")

        if last_error:
            raise APICallError(f"API 错误: {last_error}")
        raise APICallError("OpenAI 兼容接口调用失败")

    async def _make_api_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload
