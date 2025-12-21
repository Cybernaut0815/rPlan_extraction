import logging
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes via tqdm.write to keep progress bars intact."""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            pass
