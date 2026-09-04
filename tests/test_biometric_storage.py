import os
import unittest
from unittest.mock import patch

import numpy as np

from biometric_storage import (
    FACE_ENCODING_SIZE,
    DatabaseSettings,
    MySqlBiometricStorage,
    deserialize_encoding,
    serialize_encoding,
)


class FakeDatabase:
    def __init__(self):
        self.rows = {}

    def connect(self, **_options):
        return FakeConnection(self)


class FakeConnection:
    def __init__(self, database):
        self.database = database

    def cursor(self):
        return FakeCursor(self.database)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


class FakeCursor:
    def __init__(self, database):
        self.database = database
        self.result = None

    def execute(self, query, parameters=None):
        statement = " ".join(query.split()).upper()
        if statement.startswith("CREATE TABLE"):
            return
        if statement.startswith("INSERT INTO"):
            student_id, encoding_data, dimension, photo_data, content_type = parameters
            self.database.rows[student_id] = (
                encoding_data,
                dimension,
                photo_data,
                content_type,
            )
            return
        if statement.startswith("SELECT ENCODING_DATA"):
            row = self.database.rows.get(parameters[0])
            self.result = None if row is None else row[:2]
            return
        if statement == "SELECT 1":
            self.result = (1,)
            return
        raise AssertionError(f"Unexpected SQL statement: {statement}")

    def fetchone(self):
        return self.result

    def close(self):
        pass


class DatabaseSettingsTests(unittest.TestCase):
    def test_reads_required_environment_variables(self):
        environment = {
            "MYSQL_HOST": "database.example.com",
            "MYSQL_PORT": "12345",
            "MYSQL_DATABASE": "defaultdb",
            "MYSQL_USER": "avnadmin",
            "MYSQL_PASSWORD": "secret",
            "MYSQL_SSL_CA": "/secrets/ca.pem",
        }

        with patch.dict(os.environ, environment, clear=True):
            settings = DatabaseSettings.from_environment()

        self.assertEqual("database.example.com", settings.host)
        self.assertEqual(12345, settings.port)
        self.assertEqual("defaultdb", settings.database)
        self.assertEqual("avnadmin", settings.username)
        self.assertEqual("secret", settings.password)
        self.assertEqual("/secrets/ca.pem", settings.ssl_ca)

    def test_rejects_missing_environment_variables(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MYSQL_HOST"):
                DatabaseSettings.from_environment()


class EncodingSerializationTests(unittest.TestCase):
    def test_round_trip_preserves_face_encoding(self):
        original = np.linspace(-0.5, 0.5, FACE_ENCODING_SIZE, dtype=np.float64)

        payload, dimension = serialize_encoding(original)
        restored = deserialize_encoding(payload, dimension)

        np.testing.assert_array_equal(original, restored)

    def test_rejects_encoding_with_wrong_dimension(self):
        with self.assertRaisesRegex(ValueError, "exactly 128"):
            serialize_encoding(np.zeros(127, dtype=np.float64))

    def test_rejects_corrupt_stored_encoding(self):
        with self.assertRaisesRegex(ValueError, "invalid byte length"):
            deserialize_encoding(b"invalid", FACE_ENCODING_SIZE)


class MySqlBiometricStorageTests(unittest.TestCase):
    def setUp(self):
        self.database = FakeDatabase()
        self.storage = MySqlBiometricStorage(
            DatabaseSettings(
                host="database.example.com",
                port=12345,
                database="defaultdb",
                username="avnadmin",
                password="secret",
            ),
            connection_factory=self.database.connect,
        )

    def test_saved_encoding_can_be_loaded(self):
        encoding = np.linspace(-1.0, 1.0, FACE_ENCODING_SIZE)

        self.storage.save(42, encoding, b"jpeg-data")

        np.testing.assert_array_equal(encoding, self.storage.load_encoding(42))

    def test_saving_same_student_replaces_existing_biometrics(self):
        first = np.zeros(FACE_ENCODING_SIZE)
        replacement = np.ones(FACE_ENCODING_SIZE)

        self.storage.save(42, first, b"first-photo")
        self.storage.save(42, replacement, b"replacement-photo")

        self.assertEqual(1, len(self.database.rows))
        np.testing.assert_array_equal(replacement, self.storage.load_encoding(42))
        self.assertEqual(b"replacement-photo", self.database.rows[42][2])

    def test_missing_student_returns_none(self):
        self.assertIsNone(self.storage.load_encoding(404))


if __name__ == "__main__":
    unittest.main()
