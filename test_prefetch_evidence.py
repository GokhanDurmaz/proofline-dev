import importlib.machinery
import importlib.util
import tempfile
import unittest
from pathlib import Path

CANDIDATE = Path("local_agent.py")

MISSING_FILE_ISSUE = (
    "src/pkg9/File9.java:1-3: source file was not found "
    "among successfully read evidence."
)
MISSING_LINES_ISSUE = (
    "src/pkg1/File1.java:4-5: these lines were not returned "
    "by read_file: [4, 5]."
)


def load_candidate():
    loader = importlib.machinery.SourceFileLoader(
        "prefetch_agent_under_test", str(CANDIDATE.resolve())
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class PrefetchEvidenceLimitTests(unittest.TestCase):
    def setUp(self):
        self.agent = load_candidate()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        for index in range(1, 13):
            package = self.root / "src" / f"pkg{index}"
            package.mkdir(parents=True)
            (package / f"File{index}.java").write_text(
                "line one\nline two\nline three\nline four\nline five\n",
                encoding="utf-8",
            )

    def tearDown(self):
        self._tmp.cleanup()

    def read_evidence_for(self, paths):
        evidence = {}
        for path in paths:
            item = self.agent.extract_read_evidence(
                self.agent.read_file(self.root, path, 1, 3)
            )
            evidence[item["path"]] = {
                "total_lines": item["total_lines"],
                "lines": set(),
                "line_text": {},
            }
            evidence[item["path"]]["lines"].update(item["lines"])
            evidence[item["path"]]["line_text"].update(item["line_text"])
        return evidence

    def filled_evidence(self):
        return self.read_evidence_for(
            [f"src/pkg{i}/File{i}.java" for i in range(1, 9)]
        )

    def test_prefetch_at_limit_does_not_add_a_new_source_file(self):
        evidence = self.filled_evidence()
        self.assertEqual(len(evidence), self.agent.MAX_SOURCE_FILES)

        added = self.agent.prefetch_missing_read_evidence(
            self.root, [MISSING_FILE_ISSUE], evidence
        )

        self.assertEqual(added, 0)
        self.assertEqual(len(evidence), self.agent.MAX_SOURCE_FILES)
        self.assertNotIn("src/pkg9/File9.java", evidence)

    def test_prefetch_at_limit_leaves_the_citation_unresolved(self):
        evidence = self.filled_evidence()

        self.agent.prefetch_missing_read_evidence(
            self.root, [MISSING_FILE_ISSUE], evidence
        )

        issues = self.agent.validate_line_references(
            "A claim about File9 (src/pkg9/File9.java:1-3)", evidence
        )
        self.assertEqual(len(issues), 1)
        self.assertIn(
            "source file was not found among successfully read evidence",
            issues[0],
        )

    def test_prefetch_below_limit_still_adds_a_new_source_file(self):
        evidence = self.read_evidence_for(["src/pkg1/File1.java"])
        self.assertLess(len(evidence), self.agent.MAX_SOURCE_FILES)

        added = self.agent.prefetch_missing_read_evidence(
            self.root, [MISSING_FILE_ISSUE], evidence
        )

        self.assertEqual(added, 3)
        self.assertIn("src/pkg9/File9.java", evidence)
        self.assertEqual(
            self.agent.validate_line_references(
                "A claim about File9 (src/pkg9/File9.java:1-3)", evidence
            ),
            [],
        )

    def test_prefetch_at_limit_may_extend_a_file_already_in_evidence(self):
        evidence = self.filled_evidence()

        added = self.agent.prefetch_missing_read_evidence(
            self.root, [MISSING_LINES_ISSUE], evidence
        )

        self.assertEqual(added, 2)
        self.assertEqual(len(evidence), self.agent.MAX_SOURCE_FILES)
        self.assertEqual(evidence["src/pkg1/File1.java"]["lines"], {1, 2, 3, 4, 5})


if __name__ == "__main__":
    unittest.main(verbosity=2)
