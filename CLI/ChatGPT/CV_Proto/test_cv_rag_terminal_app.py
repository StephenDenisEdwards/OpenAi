import unittest
from cv_rag_terminal_app import chunk_text

class TestChunkText(unittest.TestCase):
    def setUp(self):
        self.sample_text = (
            "CV: Candidate A\n"
            "Experience in Python\n"
            "Experience in Java\n\n"
            "Additional notes\n"
            "CV: Candidate B\n"
            "Experience in C#\n"
            "Experience in SQL\n"
        )

    def test_max_tokens_small_value(self):
        # Using max_tokens=10, expecting 3 chunks.
        chunks = chunk_text(self.sample_text, max_tokens=10)
        self.assertEqual(len(chunks), 3, f"Expected 3 chunks, got {len(chunks)}: {chunks}")
        self.assertTrue(chunks[0].startswith("CV: Candidate A"), f"Chunk 0 should start with 'CV: Candidate A': {chunks[0]}")
        self.assertFalse(chunks[1].startswith("CV:"), f"Chunk 1 should be an orphan and not start with 'CV:', got: {chunks[1]}")
        self.assertTrue(chunks[2].startswith("CV: Candidate B"), f"Chunk 2 should start with 'CV: Candidate B': {chunks[2]}")

    def test_max_tokens_large_value(self):
        # Using max_tokens=100, expecting paragraphs between candidate headers to merge into 2 chunks.
        chunks = chunk_text(self.sample_text, max_tokens=100)
        self.assertEqual(len(chunks), 2, f"Expected 2 chunks, got {len(chunks)}: {chunks}")
        self.assertTrue(chunks[0].startswith("CV: Candidate A"), f"Chunk 0 should start with 'CV: Candidate A': {chunks[0]}")
        self.assertTrue(chunks[1].startswith("CV: Candidate B"), f"Chunk 1 should start with 'CV: Candidate B': {chunks[1]}")

    def test_max_tokens_five_value(self):
        # Using max_tokens=5, simulate splitting based on word count.
        # Expected chunks based on the algorithm:
        #   0: "CV: Candidate A"                        -> 3 tokens
        #   1: "Experience in Python"                   -> 3 tokens (new chunk)
        #   2: "Experience in Java\nAdditional notes"   -> 3+2 tokens = 5 tokens merged
        #   3: "CV: Candidate B"                        -> 3 tokens
        #   4: "Experience in C#"                       -> 3 tokens (new chunk)
        #   5: "Experience in SQL"                      -> 3 tokens (new chunk)
        chunks = chunk_text(self.sample_text, max_tokens=5)
        self.assertEqual(len(chunks), 6, f"Expected 6 chunks for max_tokens=5, got {len(chunks)}: {chunks}")
        self.assertTrue(chunks[0].startswith("CV: Candidate A"), f"Chunk 0 should start with 'CV: Candidate A': {chunks[0]}")
        self.assertFalse(chunks[1].startswith("CV:"), f"Chunk 1 should not start with 'CV:': {chunks[1]}")
        self.assertFalse(chunks[2].startswith("CV:"), f"Chunk 2 should not start with 'CV:': {chunks[2]}")
        self.assertTrue(chunks[3].startswith("CV: Candidate B"), f"Chunk 3 should start with 'CV: Candidate B': {chunks[3]}")
        self.assertFalse(chunks[4].startswith("CV:"), f"Chunk 4 should not start with 'CV:': {chunks[4]}")
        self.assertFalse(chunks[5].startswith("CV:"), f"Chunk 5 should not start with 'CV:': {chunks[5]}")

if __name__ == "__main__":
    unittest.main(exit=False)