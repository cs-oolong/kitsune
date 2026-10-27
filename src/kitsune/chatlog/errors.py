class ChatError(Exception):
    """An error to show the caller as JSON instead of a traceback."""

    def __init__(self, message: str, code: str = "bad_request"):
        super().__init__(message)
        self.code = code
