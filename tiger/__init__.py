from typing import Optional

__version__ = "6.3.0"


class TigerException(Exception):
    """Base class for exceptions with a dedicated message field"""

    def __init__(self, message: Optional[str] = None, code: Optional[int] = None, *args):
        super().__init__(message, code, *args)

    @property
    def message(self) -> Optional[str]:
        """Message associated with the error (can be None)"""
        return self.args[0]

    @property
    def code(self) -> Optional[int]:
        """Error code (can be None)"""
        return self.args[1]
