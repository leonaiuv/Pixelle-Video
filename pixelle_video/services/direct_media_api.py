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
        provider = image_config.get("provider", "openai_images")

        if provider != "openai_images":
            raise ValueError(f"Unsupported direct image API provider: {provider}")

        return await self._generate_openai_image(prompt=prompt, width=width, height=height, **params)

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
