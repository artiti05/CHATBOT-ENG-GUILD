import time
import uuid


class RequestTracer:
    """Tracks unique request context and request IDs across all pipeline stages."""

    @staticmethod
    def generate_request_id() -> str:
        timestamp = time.strftime("%Y%m%d")
        unique_hash = uuid.uuid4().hex[:6]
        return f"{timestamp}-{unique_hash}"
