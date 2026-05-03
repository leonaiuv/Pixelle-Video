# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Direct media API providers.

This module lets Pixelle-Video call pay-as-you-go model APIs for simple image
generation while keeping the existing ComfyUI/RunningHub workflow path intact.
"""

import base64
import os
import uuid
from typing import Any, Optional

import httpx
from loguru import logger
from openai import AsyncOpenAI

from pixelle_video.models.media import MediaResult
from pixelle_video.utils.os_util import get_temp_path, save_bytes_to_file


class DirectMediaApiService:
    """Direct image/video model API integration."""

    def __init__(self, config: dict):
        self.config = config.get("direct_media_api", {})

    @property
    def image_config(self) -> dict:
        return self.config.get("image", {})

    def is_image_enabled(self) -> bool:
        image_config = self.image_config
        return bool(image_config.get("enabled"))

    async def generate_image(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **params: Any,
    ) -> MediaResult:
        """Generate an image through the configured direct API provider."""
        image_config = self.image_config
        provider = image_config.get("provider", "openrouter_chat")

        if provider == "openrouter_chat":
            return await self._generate_openrouter_image(prompt=prompt, width=width, height=height, **params)

        if provider == "openai_images":
            return await self._generate_openai_image(prompt=prompt, width=width, height=height, **params)

        raise ValueError(f"Unsupported direct image API provider: {provider}")

    async def _generate_openrouter_image(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **params: Any,
    ) -> MediaResult:
        image_config = self.image_config
        api_key = image_config.get("api_key") or os.getenv("OPENROUTER_API_KEY") or os.getenv("PIXELLE_IMAGE_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key is required")

        base_url = (
            image_config.get("base_url")
            or os.getenv("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        ).rstrip("/")
        model = image_config.get("model") or os.getenv("PIXELLE_IMAGE_MODEL") or "openai/gpt-5.4-image-2"
        image_config_payload = self._openrouter_image_config(image_config, width, height)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/leonaiuv/Pixelle-Video",
            "X-Title": "Pixelle-Video",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["image", "text"],
            "stream": False,
        }
        if image_config_payload:
            payload["image_config"] = image_config_payload

        logger.info(f"Executing direct image API provider=openrouter_chat model={model}")
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=30.0)) as client:
            response = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

        image_urls = self._extract_openrouter_image_urls(result)
        if not image_urls:
            raise ValueError("OpenRouter image response did not include message.images")

        image_path = self._save_data_url(image_urls[0], image_config.get("output_format") or "png")
        logger.info(f"✅ OpenRouter generated image: {image_path}")
        return MediaResult(media_type="image", url=image_path)

    async def _generate_openai_image(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **params: Any,
    ) -> MediaResult:
        image_config = self.image_config
        api_key = image_config.get("api_key") or os.getenv("PIXELLE_IMAGE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Direct image API key is required")

        base_url = image_config.get("base_url") or os.getenv("PIXELLE_IMAGE_BASE_URL")
        model = image_config.get("model") or os.getenv("PIXELLE_IMAGE_MODEL") or "gpt-image-1"
        size = image_config.get("size") or self._size_from_dimensions(width, height)
        quality = image_config.get("quality") or "auto"
        output_format = image_config.get("output_format") or "png"

        client = AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key)
        logger.info(f"Executing direct image API provider=openai_images model={model} size={size}")

        response = await client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            output_format=output_format,
            n=1,
        )

        if not response.data:
            raise ValueError("Direct image API returned no image data")

        image = response.data[0]
        if image.b64_json:
            image_bytes = base64.b64decode(image.b64_json)
            suffix = self._suffix_for_format(output_format)
            image_path = get_temp_path("direct_media_api", f"{uuid.uuid4().hex}{suffix}")
            save_bytes_to_file(image_bytes, image_path)
            logger.info(f"✅ Direct image API generated image: {image_path}")
            return MediaResult(media_type="image", url=image_path)

        if image.url:
            logger.info(f"✅ Direct image API generated image URL: {image.url}")
            return MediaResult(media_type="image", url=image.url)

        raise ValueError("Direct image API returned neither b64_json nor URL")

    def _openrouter_image_config(
        self,
        image_config: dict,
        width: Optional[int],
        height: Optional[int],
    ) -> dict:
        config: dict[str, str] = {}
        aspect_ratio = self._aspect_ratio_from_size(image_config.get("size"), width, height)
        if aspect_ratio:
            config["aspect_ratio"] = aspect_ratio

        quality = image_config.get("quality")
        if quality and quality != "auto":
            config["image_size"] = {"low": "1K", "medium": "2K", "high": "4K"}.get(quality, quality)

        return config

    def _extract_openrouter_image_urls(self, result: dict) -> list[str]:
        images: list[str] = []
        for choice in result.get("choices", []):
            message = choice.get("message", {})
            for image in message.get("images") or []:
                image_url = image.get("image_url") or {}
                url = image_url.get("url")
                if url:
                    images.append(url)
        return images

    def _save_data_url(self, data_url: str, output_format: str) -> str:
        if not data_url.startswith("data:"):
            return data_url

        _, encoded = data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image_path = get_temp_path("direct_media_api", f"{uuid.uuid4().hex}{self._suffix_for_format(output_format)}")
        return save_bytes_to_file(image_bytes, image_path)

    def _aspect_ratio_from_size(
        self,
        size: Optional[str],
        width: Optional[int],
        height: Optional[int],
    ) -> Optional[str]:
        if size and size != "auto" and "x" in size:
            raw_width, raw_height = size.split("x", 1)
            try:
                width = int(raw_width)
                height = int(raw_height)
            except ValueError:
                return None

        if not width or not height:
            return None

        ratio = width / height
        known = {
            "1:1": 1.0,
            "2:3": 2 / 3,
            "3:2": 3 / 2,
            "3:4": 3 / 4,
            "4:3": 4 / 3,
            "4:5": 4 / 5,
            "5:4": 5 / 4,
            "9:16": 9 / 16,
            "16:9": 16 / 9,
        }
        return min(known, key=lambda aspect: abs(known[aspect] - ratio))

    def _size_from_dimensions(self, width: Optional[int], height: Optional[int]) -> str:
        if not width or not height:
            return "auto"

        if width == height:
            return "1024x1024"

        return "1024x1536" if height > width else "1536x1024"

    def _suffix_for_format(self, output_format: str) -> str:
        clean_format = (output_format or "png").lower()
        if clean_format in {"jpeg", "jpg"}:
            return ".jpg"
        if clean_format == "webp":
            return ".webp"
        return ".png"
