from fastapi.encoders import jsonable_encoder

SENSITIVE_KEYS = {"api_key", "apikey", "authorization", "password", "secret", "x-api-key"}


def _redact(value):
    if isinstance(value, dict):
        return {k: "[redacted]" if str(k).lower() in SENSITIVE_KEYS else _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def to_jsonable(value):
    """JSON-safe copy of state/payload data (pydantic models, exceptions, tuples) without credentials."""
    encoded = jsonable_encoder(value, custom_encoder={BaseException: lambda e: f"{type(e).__name__}: {e}"})
    return _redact(encoded)
