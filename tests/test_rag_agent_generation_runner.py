"""Make the pytest-style RAG checks runnable with the repository unittest runner."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from tests import test_rag_agent_generation as checks


class MonkeyPatchAdapter:
    def __init__(self):
        self.undo_actions = []

    def delenv(self, name, raising=True):
        old = os.environ.pop(name, None)
        self.undo_actions.append(("env", name, old))
        if raising and old is None:
            raise KeyError(name)

    def setenv(self, name, value):
        old = os.environ.get(name)
        self.undo_actions.append(("env", name, old))
        os.environ[name] = value

    def setattr(self, obj, name, value):
        old = getattr(obj, name)
        self.undo_actions.append(("attr", obj, name, old))
        setattr(obj, name, value)

    def undo(self):
        for action in reversed(self.undo_actions):
            if action[0] == "env":
                _, name, old = action
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old
            else:
                _, obj, name, old = action
                setattr(obj, name, old)


class RagAgentGenerationTests(unittest.TestCase):
    def invoke(self, function, *, with_monkeypatch=False, with_tempdir=False):
        patch = MonkeyPatchAdapter()
        try:
            if with_tempdir:
                with tempfile.TemporaryDirectory() as folder:
                    function(Path(folder), patch)
            elif with_monkeypatch:
                function(patch)
            else:
                function()
        finally:
            patch.undo()

    def test_category_routing_and_citations(self):
        self.invoke(checks.test_chat_routes_each_category_and_preserves_citations, with_monkeypatch=True)

    def test_uncertain_diagnosis_context(self):
        self.invoke(checks.test_low_confidence_context_is_not_confirmed_and_is_retrieval_context)

    def test_empty_and_unavailable_retrieval(self):
        self.invoke(checks.test_empty_and_unavailable_retrieval_are_honest)

    def test_disable_options_do_not_emit_fake_events(self):
        self.invoke(checks.test_specialist_and_rag_can_be_disabled_without_fake_events)

    def test_provider_error_falls_back_with_error_status(self):
        self.invoke(checks.test_configured_provider_failure_falls_back_and_reports_error, with_monkeypatch=True)

    def test_local_vector_index_persistence(self):
        self.invoke(checks.test_local_vector_store_persists_and_reloads, with_monkeypatch=True, with_tempdir=True)


if __name__ == "__main__":
    unittest.main()
