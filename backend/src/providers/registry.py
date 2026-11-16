from __future__ import annotations

import os
from pathlib import Path

import yaml

from src import usage
from src.logger import get_logger
from src.providers.base import TOOL_MODES, BaseProvider, ModelOption

_log = get_logger(__name__)

CATALOG_PATH = Path(__file__).resolve().parents[2] / "models.yaml"


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: list[BaseProvider] = []
        self._model_map: dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        self._providers.append(provider)
        for model in provider.models:
            self._model_map.setdefault(model.id, provider)
            usage.set_price(model.id, model.input_price, model.output_price)

    @property
    def providers(self) -> list[BaseProvider]:
        return list(self._providers)

    def list_models(self) -> list[ModelOption]:
        return [m for p in self._providers for m in p.models]

    def get_provider(self, model_id: str) -> BaseProvider:
        if model_id not in self._model_map:
            raise KeyError(f"Unknown model: {model_id}")
        return self._model_map[model_id]

    def get_model(self, model_id: str) -> ModelOption:
        model = self.get_provider(model_id).model(model_id)
        if model is None:
            raise KeyError(f"Unknown model: {model_id}")
        return model

    def _automatic(self) -> list[BaseProvider]:
        """Providers the app may pick on its own. The in-process local model is only used when chosen
        explicitly (model menu, or the utility-model setting), or when nothing else is configured."""
        others = [p for p in self._providers if p.provider_name != "local"]
        return others or list(self._providers)

    @property
    def default_model_id(self) -> str:
        """Model for requests that don't name one: the first ready automatic model."""
        candidates = self._automatic()
        ready = [p for p in candidates if p.ready] or candidates
        return ready[0].models[0].id

    @property
    def cheapest_model(self) -> ModelOption:
        """Model for utility calls (titles, memory, routing): the first ready is_cheapest model in
        registration order, then the cheapest ready model of any kind, then the first flagged model,
        then the default. The local model is skipped unless nothing else is configured.
        """
        providers = self._automatic()
        flagged = [(p, m) for p in providers for m in p.models if m.is_cheapest]
        for provider, model in flagged:
            if provider.ready:
                return model
        # No flagged model can serve right now: use the cheapest one that can (e.g. an Ollama model).
        ready = [m for p in providers if p.ready for m in p.models]
        if ready:
            return min(ready, key=lambda m: m.input_price + m.output_price)
        if flagged:
            return flagged[0][1]
        return providers[0].models[0]


# ── catalog loading ───────────────────────────────────────────────────────

_MODEL_FIELDS = set(ModelOption.__dataclass_fields__) - {"provider"}


def _model_option(raw: dict | str, provider: str, defaults: dict) -> ModelOption:
    spec = {"id": raw} if isinstance(raw, str) else dict(raw)
    merged = {**defaults, **spec}
    unknown = set(merged) - _MODEL_FIELDS
    if unknown:
        _log.warning("models.yaml: ignoring unknown field(s) %s on %s", sorted(unknown), merged.get("id"))
    opts = {k: v for k, v in merged.items() if k in _MODEL_FIELDS}
    opts.setdefault("name", opts["id"])
    if opts.get("tools", "auto") not in TOOL_MODES:
        raise ValueError(f"models.yaml: {opts['id']}: tools must be one of {TOOL_MODES}")
    return ModelOption(provider=provider, **opts)


def load_catalog(path: Path | str | None = None) -> dict:
    p = Path(path) if path else CATALOG_PATH
    if not p.exists():
        return {"providers": {"local": {"type": "builtin"}}}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def build_provider(name: str, spec: dict, llm=None) -> BaseProvider | None:
    """Build one provider from its catalog entry, or None when it can't be used (missing key)."""
    kind = spec.get("type", "openai_compatible")
    if spec.get("enabled") is False:
        return None
    key_env = spec.get("api_key_env")
    api_key = os.getenv(key_env) if key_env else None
    if key_env and not api_key:
        return None
    defaults = dict(spec.get("defaults") or {})
    models = [_model_option(m, name, defaults) for m in spec.get("models") or []]

    if kind == "builtin":
        if llm is None:
            return None
        from src.providers.local import LocalProvider
        return LocalProvider(llm)
    if kind == "anthropic":
        from src.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key, models=models, discover=bool(spec.get("discover")),
                                 discovered_defaults=defaults)
    if kind in ("openai", "openai_compatible"):
        from src.providers.openai_provider import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider(
            name, api_key=api_key, base_url=spec.get("base_url"), models=models,
            discover=bool(spec.get("discover")), discovered_defaults=defaults,
        )
        return provider if provider.models else None
    raise ValueError(f"models.yaml: provider {name!r} has unknown type {kind!r}")


def check_api_keys(registry: ProviderRegistry) -> None:
    """Verify each provider's API key (a model listing, no tokens), so a bad key is flagged before first use."""
    for provider in registry.providers:
        try:
            provider.check_auth()
        except Exception as exc:  # a check must never take the app down
            _log.info("API key check failed for %s: %s", provider.provider_name, exc)


def build_registry(llm, catalog_path: Path | str | None = None) -> ProviderRegistry:
    """Register every usable provider from models.yaml, in file order."""
    registry = ProviderRegistry()
    for name, spec in (load_catalog(catalog_path).get("providers") or {}).items():
        try:
            provider = build_provider(name, spec or {}, llm)
        except Exception as exc:
            _log.error("Could not set up provider %s: %s", name, exc)
            continue
        if provider is not None and provider.models:
            provider.api_key_env = (spec or {}).get("api_key_env")
            registry.register(provider)
            _log.info("Provider %s: %s", name, ", ".join(m.id for m in provider.models))
    if not registry.providers:
        raise RuntimeError("No models available. Add an API key to backend/.env, start Ollama or LM Studio, "
                           "or enable the local model (model.enabled: true in config.yaml).")
    return registry
