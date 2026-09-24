import re
import sys

# A base64 blob in the logs is always an image payload: Gradio embeds the frame it
# received into input-validation errors, and `cam.stream` retries twice a second,
# so one stale browser tab can bury the terminal under megabytes of JPEG.
_BLOB = re.compile(r"(?:data:[\w./+-]+;base64,)?[A-Za-z0-9+/]{200,}={0,2}")


class _EllipsizingStream:
    """Passthrough stream that replaces base64 payloads with their size."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, text: str) -> int:
        return self._stream.write(
            _BLOB.sub(lambda m: f"<base64 blob, {len(m.group(0))} chars elided>", text)
        )

    def __getattr__(self, name):
        return getattr(self._stream, name)


def elide_base64_in_logs() -> None:
    sys.stdout = _EllipsizingStream(sys.stdout)
    sys.stderr = _EllipsizingStream(sys.stderr)
