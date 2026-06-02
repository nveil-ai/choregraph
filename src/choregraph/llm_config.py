# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Clément Baraille
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Provider-aware LLM configuration for choregraph's LLM-assisted nodes.

Both CSV characterization (``loaders.py``) and Excel tidying
(``collection/excel``) need a chat model whose model name and
provider-native kwargs come from the per-provider config YAMLs the
ai_service owns (``llm_processing/configs/<provider>.yaml``). Those files
are the single source of truth — nothing is hardcoded here:

* CSV characterization uses the ``minimal:`` profile (cheap model).
* Excel tidying uses the ``defaults:`` profile (main model).

choregraph sits *below* the backend in the dependency graph and must not
import ``shared`` / ``ai_service``; it reads those YAMLs at runtime
instead. The provider tables below are KEEP-IN-SYNC duplicates of
``nveil/backend/shared/llm_config.py`` (a separate package can't share the
constants directly).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider tables — KEEP IN SYNC with nveil/backend/shared/llm_config.py.
# ---------------------------------------------------------------------------

# Boot order walked to auto-detect the active provider. There is deliberately
# no LLM_PROVIDER env var: selection mirrors `LLMConfig.from_env_ordered()` —
# the first provider whose key (or endpoint+model for locals) is configured.
PROVIDER_BOOT_ORDER: tuple = (
    "google_genai", "openai", "anthropic", "mistralai",
    "ollama", "llamacpp", "openai_compatible",
)

PROVIDER_ENV_KEY: Dict[str, str] = {
    "google_genai": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    # Ollama / llama.cpp are OpenAI-compatible; reuse OPENAI_API_KEY (any
    # dummy value — these endpoints don't authenticate by default).
    "ollama": "OPENAI_API_KEY",
    "llamacpp": "OPENAI_API_KEY",
    "openai_compatible": "OPENAI_COMPAT_API_KEY",
}

# Provider name -> LangChain `model_provider` string. Unlisted providers use
# their own name. Ollama/llama.cpp/openai_compatible expose OpenAI APIs.
LANGCHAIN_PROVIDER: Dict[str, str] = {
    "ollama": "openai", "llamacpp": "openai", "openai_compatible": "openai",
}

# Local providers that don't authenticate: they need base_url + model but the
# api_key is an opaque non-empty sentinel (ChatOpenAI requires a string).
LOCAL_PROVIDERS_WITHOUT_KEY: frozenset = frozenset({"ollama", "llamacpp"})
LOCAL_API_KEY_SENTINEL: str = "local-provider-no-auth"


# ---------------------------------------------------------------------------
# Config YAML location + parsing
# ---------------------------------------------------------------------------

_yaml_cache: Dict[str, Optional[dict]] = {}


def configs_dir() -> Optional[Path]:
    """Locate the ai_service provider config directory.

    Read at runtime (no import of ai_service code). Resolution order:
    1. ``LLM_CONFIGS_DIR`` env var.
    2. Repo-relative path (works for ``pip install -e`` and the co-mounted
       container layout).
    3. The canonical absolute path inside the deployment image.
    """
    candidates = []
    env_dir = os.getenv("LLM_CONFIGS_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(
        Path(__file__).resolve().parents[3]
        / "nveil" / "backend" / "ai_service" / "llm_processing" / "configs"
    )
    candidates.append(Path("/nveil/backend/ai_service/llm_processing/configs"))
    for c in candidates:
        if c.is_dir():
            return c
    return None


def load_provider_yaml(provider: str) -> Optional[dict]:
    """Parse ``<configs_dir>/<provider>.yaml`` (cached). Returns ``None``
    when the configs can't be located / parsed (callers then skip the LLM)."""
    if provider in _yaml_cache:
        return _yaml_cache[provider]

    result: Optional[dict] = None
    cfg_dir = configs_dir()
    if cfg_dir is None:
        logger.warning("load_provider_yaml: configs dir not found — set LLM_CONFIGS_DIR")
    else:
        path = cfg_dir / f"{provider}.yaml"
        if path.is_file():
            try:
                import yaml

                with open(path) as f:
                    result = yaml.safe_load(f) or {}
            except Exception as exc:
                logger.warning("load_provider_yaml: could not read %s: %s", path, exc)
    _yaml_cache[provider] = result
    return result


def resolve_profile(provider: str, profile: str) -> Optional[dict]:
    """Resolve a provider's ``{model, **native_kwargs}`` for *profile*.

    ``defaults`` is the base; for any other profile (e.g. ``minimal``) the
    named top-level block is layered on top, where a ``null`` value removes
    the key — the same merge semantics as the ai_service's
    ``node_config.get_minimal_config``. Returns ``None`` when there is no
    usable model.
    """
    cfg = load_provider_yaml(provider)
    if cfg is None:
        return None
    merged: Dict[str, Any] = dict(cfg.get("defaults", {}) or {})
    if profile != "defaults":
        for k, v in (cfg.get(profile, {}) or {}).items():
            if v is None:
                merged.pop(k, None)
            else:
                merged[k] = v
    if "model" not in merged:
        return None
    return merged


# ---------------------------------------------------------------------------
# Credential + provider selection
# ---------------------------------------------------------------------------


def resolve_api_key(api_key: Optional[str] = None, provider: str = "google_genai") -> Optional[str]:
    """Resolve the API key for *provider*.

    Resolution order: explicit *api_key* arg → provider env var →
    ``.env`` discovered by python-dotenv.
    """
    if api_key:
        return api_key

    env_var = PROVIDER_ENV_KEY.get(provider, "GOOGLE_API_KEY")
    key = os.getenv(env_var)
    if key:
        return key

    try:
        from dotenv import find_dotenv, load_dotenv

        env_found = find_dotenv(usecwd=True)
        if env_found:
            load_dotenv(env_found, override=False)
            key = os.getenv(env_var)
            if key:
                return key
    except ImportError:
        pass

    return None


def select_provider() -> Optional[dict]:
    """Auto-detect the active provider, mirroring `from_env_ordered()`.

    Walks ``PROVIDER_BOOT_ORDER`` over the env vars and returns the first
    usable provider as ``{provider, api_key, base_url, model_override}`` —
    or ``None`` when none is configured (the LLM step is then skipped).
    """
    for provider in PROVIDER_BOOT_ORDER:
        if provider == "openai_compatible":
            base_url = os.getenv("OPENAI_COMPAT_BASE_URL")
            api_key = os.getenv("OPENAI_COMPAT_API_KEY")
            model = os.getenv("OPENAI_COMPAT_MODEL")
            if base_url and api_key and model:
                return {"provider": provider, "api_key": api_key,
                        "base_url": base_url, "model_override": model}
            continue

        if provider in LOCAL_PROVIDERS_WITHOUT_KEY:
            base_url = os.getenv(f"{provider.upper()}_BASE_URL")
            model = os.getenv(f"{provider.upper()}_MODEL")
            if base_url and model:
                return {"provider": provider, "api_key": LOCAL_API_KEY_SENTINEL,
                        "base_url": base_url, "model_override": model}
            continue

        api_key = resolve_api_key(provider=provider)
        if api_key:
            return {"provider": provider, "api_key": api_key,
                    "base_url": None, "model_override": None}
    return None


# ---------------------------------------------------------------------------
# Chat model builder
# ---------------------------------------------------------------------------


def build_chat_model(
    provider: str,
    api_key: str,
    profile: str,
    *,
    base_url: Optional[str] = None,
    model_override: Optional[str] = None,
    max_retries: int = 2,
):
    """Instantiate a LangChain chat model for *provider* using *profile*.

    Model name and provider-native kwargs come from the provider YAML
    (``defaults`` for Excel, ``minimal`` for CSV). ``model_override`` wins
    over the yaml model (local/custom endpoints carry the real model name
    in env). Raises ``ValueError`` if the profile can't be resolved.
    """
    cfg = resolve_profile(provider, profile)
    if cfg is None:
        raise ValueError(
            f"No '{profile}' config for provider {provider!r} — "
            "check the provider yaml / LLM_CONFIGS_DIR."
        )
    model = model_override or cfg.get("model")
    # native kwargs (thinking_level, response_format, …); a null value means
    # "disable", so drop it rather than pass None.
    native_kwargs = {k: v for k, v in cfg.items() if k != "model" and v is not None}

    from langchain.chat_models import init_chat_model

    init_kwargs: Dict[str, Any] = {
        "model": model,
        "model_provider": LANGCHAIN_PROVIDER.get(provider, provider),
        "api_key": api_key,
        "max_retries": max_retries,
        **native_kwargs,
    }
    if base_url:
        init_kwargs["base_url"] = base_url
    return init_chat_model(**init_kwargs)
