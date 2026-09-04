import os
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


FACE_ENCODING_SIZE = 128


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_ca: Optional[str] = None

    @classmethod
    def from_environment(cls) -> "DatabaseSettings":
        variable_names = {
            "host": "MYSQL_HOST",
            "database": "MYSQL_DATABASE",
            "username": "MYSQL_USER",
            "password": "MYSQL_PASSWORD",
        }
        values = {field: os.getenv(name) for field, name in variable_names.items()}
        missing = [name for field, name in variable_names.items() if not values[field]]
        if missing:
            raise RuntimeError(
                "Missing required database environment variables: " + ", ".join(missing)
            )

        try:
            port = int(os.getenv("MYSQL_PORT", "3306"))
        except ValueError as error:
            raise RuntimeError("MYSQL_PORT must be a valid integer") from error

        if not 1 <= port <= 65535:
            raise RuntimeError("MYSQL_PORT must be between 1 and 65535")

        return cls(
            host=values["host"],
            port=port,
            database=values["database"],
            username=values["username"],
            password=values["password"],
            ssl_ca=os.getenv("MYSQL_SSL_CA") or None,
        )


def serialize_encoding(encoding: np.ndarray) -> Tuple[bytes, int]:
    vector = np.asarray(encoding, dtype=np.float64)
    if vector.ndim != 1 or vector.size != FACE_ENCODING_SIZE:
        raise ValueError(f"Face encoding must contain exactly {FACE_ENCODING_SIZE} values")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Face encoding contains invalid numeric values")
    return vector.tobytes(order="C"), int(vector.size)


def deserialize_encoding(payload: bytes, dimension: int) -> np.ndarray:
    if dimension != FACE_ENCODING_SIZE:
        raise ValueError(f"Stored face encoding has an invalid dimension: {dimension}")

    expected_bytes = dimension * np.dtype(np.float64).itemsize
    if len(payload) != expected_bytes:
        raise ValueError("Stored face encoding has an invalid byte length")

    vector = np.frombuffer(payload, dtype=np.float64).copy()
    if not np.all(np.isfinite(vector)):
        raise ValueError("Stored face encoding contains invalid numeric values")
    return vector


class MySqlBiometricStorage:
    TABLE_NAME = "face_biometrics"

    def __init__(
        self,
        settings: DatabaseSettings,
        connection_factory: Optional[Callable[..., object]] = None,
    ):
        self.settings = settings
        self.connection_factory = connection_factory
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    @classmethod
    def from_environment(cls) -> "MySqlBiometricStorage":
        return cls(DatabaseSettings.from_environment())

    def _connect(self):
        if self.connection_factory is None:
            import mysql.connector

            connection_factory = mysql.connector.connect
        else:
            connection_factory = self.connection_factory

        connection_options = {
            "host": self.settings.host,
            "port": self.settings.port,
            "database": self.settings.database,
            "user": self.settings.username,
            "password": self.settings.password,
            "ssl_disabled": False,
            "connection_timeout": 10,
        }
        if self.settings.ssl_ca:
            connection_options.update(
                {
                    "ssl_ca": self.settings.ssl_ca,
                    "ssl_verify_cert": True,
                    "ssl_verify_identity": True,
                }
            )
        return connection_factory(**connection_options)

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return

        with self._schema_lock:
            if self._schema_ready:
                return

            connection = self._connect()
            cursor = connection.cursor()
            try:
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                        student_id BIGINT NOT NULL PRIMARY KEY,
                        encoding_data BLOB NOT NULL,
                        encoding_dimension SMALLINT UNSIGNED NOT NULL,
                        photo_data LONGBLOB NOT NULL,
                        photo_content_type VARCHAR(64) NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP
                    )
                    """
                )
                connection.commit()
                self._schema_ready = True
            finally:
                cursor.close()
                connection.close()

    def save(
        self,
        student_id: int,
        encoding: np.ndarray,
        photo_data: bytes,
        photo_content_type: str = "image/jpeg",
    ) -> None:
        if student_id <= 0:
            raise ValueError("Student ID must be positive")
        if not photo_data:
            raise ValueError("Registration photo cannot be empty")

        encoding_data, dimension = serialize_encoding(encoding)
        self._ensure_schema()

        connection = self._connect()
        cursor = connection.cursor()
        try:
            cursor.execute(
                f"""
                INSERT INTO {self.TABLE_NAME}
                    (student_id, encoding_data, encoding_dimension, photo_data, photo_content_type)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    encoding_data = VALUES(encoding_data),
                    encoding_dimension = VALUES(encoding_dimension),
                    photo_data = VALUES(photo_data),
                    photo_content_type = VALUES(photo_content_type)
                """,
                (student_id, encoding_data, dimension, photo_data, photo_content_type),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def load_encoding(self, student_id: int) -> Optional[np.ndarray]:
        if student_id <= 0:
            return None

        self._ensure_schema()
        connection = self._connect()
        cursor = connection.cursor()
        try:
            cursor.execute(
                f"""
                SELECT encoding_data, encoding_dimension
                FROM {self.TABLE_NAME}
                WHERE student_id = %s
                """,
                (student_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return deserialize_encoding(bytes(row[0]), int(row[1]))
        finally:
            cursor.close()
            connection.close()

    def ping(self) -> None:
        self._ensure_schema()
        connection = self._connect()
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        finally:
            cursor.close()
            connection.close()
