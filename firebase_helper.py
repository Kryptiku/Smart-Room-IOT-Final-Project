import os
import time
from pathlib import Path
from typing import Any

import firebase_admin
from firebase_admin import credentials, db
from dotenv import load_dotenv
_firebase_app: Any | None = None
_firebase_disabled_reason: str | None = None

# Load env from the standard root files first, then fall back to the frontend
# env used by the dashboard so the camera can reuse the same RTDB URL.
load_dotenv()
load_dotenv(Path(".env.example"), override=False)
load_dotenv(Path(".env.local"), override=False)
load_dotenv(Path("frontend") / ".env.local", override=False)


def _find_credentials_path() -> str | None:
    explicit_path = os.getenv("FIREBASE_CREDENTIALS_PATH") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit_path:
        return explicit_path

    fallback_path = Path("serviceAccountKey.json")
    if fallback_path.exists():
        return str(fallback_path)

    return None


def _find_database_url() -> str | None:
    return os.getenv("FIREBASE_DATABASE_URL") or os.getenv("NEXT_PUBLIC_FIREBASE_DATABASE_URL")


def firebase_is_configured() -> bool:
    # Writing to a locked-down Realtime Database requires a service account.
    return _find_credentials_path() is not None and _find_database_url() is not None


def init_firebase() -> Any:
    global _firebase_app, _firebase_disabled_reason

    if _firebase_app is not None:
        return _firebase_app

    if _firebase_disabled_reason is not None:
        raise RuntimeError(_firebase_disabled_reason)

    credentials_path = _find_credentials_path()
    database_url = _find_database_url()

    if not database_url:
        _firebase_disabled_reason = "FIREBASE_DATABASE_URL is required to write to Firebase"
        raise RuntimeError(_firebase_disabled_reason)

    if not credentials_path:
        _firebase_disabled_reason = (
            "FIREBASE_CREDENTIALS_PATH or GOOGLE_APPLICATION_CREDENTIALS is required to write when .write is false"
        )
        raise RuntimeError(_firebase_disabled_reason)

    firebase_credentials = credentials.Certificate(credentials_path)
    _firebase_app = firebase_admin.initialize_app(firebase_credentials, {
        "databaseURL": database_url,
    })
    return _firebase_app


def publish_room_state(occupancy_count: int, threshold: int = 5) -> dict[str, Any]:
    # Ensure we at least have a database URL to target
    database_url = _find_database_url()
    if not database_url:
        raise RuntimeError("FIREBASE_DATABASE_URL is required to write to Firebase")

    aircon_on = occupancy_count >= threshold
    payload = {
        "occupancy": occupancy_count,
        "airconOn": aircon_on,
        "airconStatus": "ON" if aircon_on else "OFF",
        "threshold": threshold,
        "updatedAt": int(time.time()),
    }

    init_firebase()
    db.reference("roomState").set(payload)

    return payload