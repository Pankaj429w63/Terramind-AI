import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.model_artifacts import resolve_model_checkpoint


class ModelArtifactTests(unittest.TestCase):
    def test_existing_checkpoint_is_used_without_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            checkpoint.write_bytes(b"unchanged")
            with patch.dict(os.environ, {"TERRAMIND_MODEL_PATH": str(checkpoint)}, clear=False), \
                 patch("backend.model_artifacts.get_supabase", create=True) as get_client:
                self.assertEqual(resolve_model_checkpoint(), checkpoint)
                get_client.assert_not_called()

    def test_private_storage_download_is_sha256_verified(self):
        content = b"model checkpoint fixture"

        class StorageBucket:
            def download(self, object_path):
                self.object_path = object_path
                return content

        bucket = StorageBucket()

        class Storage:
            def from_(self, name):
                self.name = name
                return bucket

        class Client:
            storage = Storage()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "restored" / "model.pth"
            env = {
                "TERRAMIND_MODEL_PATH": str(checkpoint),
                "TERRAMIND_MODEL_BUCKET": "model-artifacts",
                "TERRAMIND_MODEL_OBJECT": "efficientnet-b0/best_model.pth",
                "TERRAMIND_MODEL_SHA256": hashlib.sha256(content).hexdigest(),
            }
            with patch.dict(os.environ, env, clear=False), patch("backend.supabase_client.get_supabase", return_value=Client()):
                self.assertEqual(resolve_model_checkpoint(), checkpoint)
            self.assertEqual(checkpoint.read_bytes(), content)

    def test_storage_download_rejects_checksum_mismatch(self):
        class Bucket:
            def download(self, object_path):
                return b"not the expected checkpoint"

        class Client:
            class Storage:
                def from_(self, name):
                    return Bucket()
            storage = Storage()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            env = {
                "TERRAMIND_MODEL_PATH": str(checkpoint),
                "TERRAMIND_MODEL_BUCKET": "model-artifacts",
                "TERRAMIND_MODEL_OBJECT": "efficientnet-b0/best_model.pth",
                "TERRAMIND_MODEL_SHA256": "0" * 64,
            }
            with patch.dict(os.environ, env, clear=False), patch("backend.supabase_client.get_supabase", return_value=Client()):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    resolve_model_checkpoint()
            self.assertFalse(checkpoint.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
