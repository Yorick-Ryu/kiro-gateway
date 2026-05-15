# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Model metadata cache for Kiro Gateway.

Thread-safe storage for available model information
with TTL and lazy loading support.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from kiro.config import (
    MODEL_CACHE_TTL,
    DEFAULT_MAX_INPUT_TOKENS,
    PROMPT_CACHE_TTL_SECONDS,
    PROMPT_CACHE_ACCOUNTING_ENABLED,
)


class ModelInfoCache:
    """
    Thread-safe cache for storing model metadata.
    
    Uses Lazy Loading for population - data is loaded
    only on first access or when cache is stale.
    
    Attributes:
        cache_ttl: Cache time-to-live in seconds
    
    Example:
        >>> cache = ModelInfoCache()
        >>> await cache.update([{"modelId": "claude-sonnet-4", "tokenLimits": {...}}])
        >>> info = cache.get("claude-sonnet-4")
        >>> max_tokens = cache.get_max_input_tokens("claude-sonnet-4")
    """
    
    def __init__(self, cache_ttl: int = MODEL_CACHE_TTL):
        """
        Initializes the model cache.
        
        Args:
            cache_ttl: Cache time-to-live in seconds (default from config)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._last_update: Optional[float] = None
        self._cache_ttl = cache_ttl
    
    async def update(self, models_data: List[Dict[str, Any]]) -> None:
        """
        Updates the model cache.
        
        Thread-safely replaces cache contents with new data.
        
        Args:
            models_data: List of dictionaries with model information.
                        Each dictionary must contain the "modelId" key.
        """
        async with self._lock:
            logger.info(f"Updating model cache. Found {len(models_data)} models.")
            self._cache = {model["modelId"]: model for model in models_data}
            self._last_update = time.time()
    
    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Returns model information.
        
        Args:
            model_id: Model ID
        
        Returns:
            Dictionary with model information or None if model not found
        """
        return self._cache.get(model_id)
    
    def is_valid_model(self, model_id: str) -> bool:
        """
        Check if model exists in dynamic cache.
        
        Used by ModelResolver to verify if a model is available.
        
        Args:
            model_id: Model ID to check
        
        Returns:
            True if model exists in cache, False otherwise
        """
        return model_id in self._cache
    
    def add_hidden_model(self, display_name: str, internal_id: str) -> None:
        """
        Add a hidden model to the cache.
        
        Hidden models are not returned by Kiro /ListAvailableModels API
        but are still functional. They are added to the cache so they
        appear in our /v1/models endpoint.
        
        Args:
            display_name: Model name to display (e.g., "claude-3.7-sonnet")
            internal_id: Internal Kiro ID (e.g., "CLAUDE_3_7_SONNET_20250219_V1_0")
        """
        if display_name not in self._cache:
            self._cache[display_name] = {
                "modelId": display_name,
                "modelName": display_name,
                "description": f"Hidden model (internal: {internal_id})",
                "tokenLimits": {"maxInputTokens": DEFAULT_MAX_INPUT_TOKENS},
                "_internal_id": internal_id,  # Store internal ID for reference
                "_is_hidden": True,  # Mark as hidden model
            }
            logger.debug(f"Added hidden model: {display_name} → {internal_id}")
    
    def get_max_input_tokens(self, model_id: str) -> int:
        """
        Returns maxInputTokens for the model.
        
        Args:
            model_id: Model ID
        
        Returns:
            Maximum number of input tokens or DEFAULT_MAX_INPUT_TOKENS
        """
        model = self._cache.get(model_id)
        if model and model.get("tokenLimits"):
            return model["tokenLimits"].get("maxInputTokens") or DEFAULT_MAX_INPUT_TOKENS
        return DEFAULT_MAX_INPUT_TOKENS
    
    def is_empty(self) -> bool:
        """
        Checks if the cache is empty.
        
        Returns:
            True if cache is empty
        """
        return not self._cache
    
    def is_stale(self) -> bool:
        """
        Checks if the cache is stale.
        
        Returns:
            True if cache is stale (more than cache_ttl seconds have passed)
            or if cache was never updated
        """
        if not self._last_update:
            return True
        return time.time() - self._last_update > self._cache_ttl
    
    def get_all_model_ids(self) -> List[str]:
        """
        Returns a list of all model IDs in the cache.
        
        Returns:
            List of model IDs
        """
        return list(self._cache.keys())
    
    @property
    def size(self) -> int:
        """Number of models in the cache."""
        return len(self._cache)
    
    @property
    def last_update_time(self) -> Optional[float]:
        """Last update time (timestamp) or None."""
        return self._last_update


@dataclass
class PromptCacheUsage:
    """Anthropic-compatible prompt cache usage fields."""

    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def to_anthropic_usage_fields(self) -> Dict[str, int]:
        """Return non-zero Anthropic usage fields."""
        fields: Dict[str, int] = {}
        if self.cache_read_input_tokens > 0:
            fields["cache_read_input_tokens"] = self.cache_read_input_tokens
        if self.cache_creation_input_tokens > 0:
            fields["cache_creation_input_tokens"] = self.cache_creation_input_tokens
        return fields


@dataclass
class _PromptCacheEntry:
    tokens: int
    expires_at: float


class PromptCacheTracker:
    """
    In-memory prompt cache accounting.

    This only reports Anthropic cache usage fields. It does not cache model
    responses and does not alter upstream requests. Entry TTL is fixed when the
    entry is created and is not refreshed by cache hits.
    """

    def __init__(
        self,
        cache_ttl: int = PROMPT_CACHE_TTL_SECONDS,
        enabled: bool = PROMPT_CACHE_ACCOUNTING_ENABLED,
    ):
        self._cache_ttl = cache_ttl
        self._enabled = enabled
        self._entries: Dict[str, _PromptCacheEntry] = {}
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def size(self) -> int:
        self._prune_expired(time.time())
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    async def record(
        self,
        *,
        model: str,
        messages: Optional[List[Dict[str, Any]]],
        tools: Optional[List[Dict[str, Any]]],
        system: Optional[Any],
        input_tokens: int,
    ) -> PromptCacheUsage:
        """
        Record a successful request and return cache accounting usage.

        Args:
            model: Requested model name
            messages: Anthropic request messages as dictionaries
            tools: Anthropic request tools as dictionaries
            system: Anthropic system prompt
            input_tokens: Estimated input tokens for this request
        """
        if not self._enabled or input_tokens <= 0:
            return PromptCacheUsage()

        key = self._fingerprint(
            model=model,
            messages=messages or [],
            tools=tools,
            system=system,
        )
        now = time.time()

        async with self._lock:
            self._prune_expired(now)
            entry = self._entries.get(key)
            if entry and entry.expires_at > now:
                return PromptCacheUsage(cache_read_input_tokens=entry.tokens)

            self._entries[key] = _PromptCacheEntry(
                tokens=input_tokens,
                expires_at=now + self._cache_ttl,
            )
            return PromptCacheUsage(cache_creation_input_tokens=input_tokens)

    def _prune_expired(self, now: float) -> None:
        expired_keys = [
            key for key, entry in self._entries.items()
            if entry.expires_at <= now
        ]
        for key in expired_keys:
            self._entries.pop(key, None)

    def _fingerprint(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        system: Optional[Any],
    ) -> str:
        payload = {
            "model": model,
            "system": self._normalize(system),
            "tools": self._normalize(tools or []),
            "messages": self._normalize(messages),
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _normalize(self, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return self._normalize(value.model_dump(exclude_none=True))
        if isinstance(value, dict):
            return {
                str(key): self._normalize(item)
                for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
                if item is not None
            }
        if isinstance(value, list):
            return [self._normalize(item) for item in value]
        return value


prompt_cache_tracker = PromptCacheTracker()
