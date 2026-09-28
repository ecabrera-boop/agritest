import logging


class SuppressBrokenPipeFilter(logging.Filter):
    """Drop noisy dev-server logs when browsers abort in-flight connections."""

    _NOISE = ("Broken pipe", "Connection reset by peer", "Connection aborted")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(fragment in message for fragment in self._NOISE)
