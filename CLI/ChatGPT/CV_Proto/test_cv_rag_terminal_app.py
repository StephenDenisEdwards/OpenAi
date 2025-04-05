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
        # Using max_tokens=10 forces a chunk break after a small token count,
        # resulting in three chunks (including an orphan chunk).
        chunks = chunk_text(self.sample_text, max_tokens=10)
        self.assertEqual(len(chunks), 3, f"Expected 3 chunks for max_tokens=10, got {len(chunks)}: {chunks}")
        self.assertTrue(chunks[0].startswith("CV: Candidate A"),
                        f"Chunk 0 should start with 'CV: Candidate A': {chunks[0]}")
        self.assertFalse(chunks[1].startswith("CV:"),
                         f"Chunk 1 should be an orphan and not start with 'CV:', got: {chunks[1]}")
        self.assertTrue(chunks[2].startswith("CV: Candidate B"),
                        f"Chunk 2 should start with 'CV: Candidate B': {chunks[2]}")

    def test_max_tokens_large_value(self):
        # Using max_tokens=100 should merge the paragraphs between candidate headers,
        # resulting in two chunks.
        chunks = chunk_text(self.sample_text, max_tokens=100)
        self.assertEqual(len(chunks), 2, f"Expected 2 chunks for max_tokens=100, got {len(chunks)}: {chunks}")
        self.assertTrue(chunks[0].startswith("CV: Candidate A"),
                        f"Chunk 0 should start with 'CV: Candidate A': {chunks[0]}")
        self.assertTrue(chunks[1].startswith("CV: Candidate B"),
                        f"Chunk 1 should start with 'CV: Candidate B': {chunks[1]}")

    def test_max_tokens_five_value(self):
        # Using max_tokens=5, simulate very aggressive splitting.
        # Expected chunks based on the algorithm:
        #   0: "CV: Candidate A"                        -> 3 tokens
        #   1: "Experience in Python"                   -> 3 tokens (new chunk)
        #   2: "Experience in Java\nAdditional notes"   -> 3+2 tokens = 5 tokens merged
        #   3: "CV: Candidate B"                        -> 3 tokens
        #   4: "Experience in C#"                       -> 3 tokens (new chunk)
        #   5: "Experience in SQL"                      -> 3 tokens (new chunk)
        chunks = chunk_text(self.sample_text, max_tokens=5)
        self.assertEqual(len(chunks), 6, f"Expected 6 chunks for max_tokens=5, got {len(chunks)}: {chunks}")
        self.assertTrue(chunks[0].startswith("CV: Candidate A"),
                        f"Chunk 0 should start with 'CV: Candidate A': {chunks[0]}")
        self.assertFalse(chunks[1].startswith("CV:"), f"Chunk 1 should not start with 'CV:', got: {chunks[1]}")
        self.assertFalse(chunks[2].startswith("CV:"), f"Chunk 2 should not start with 'CV:', got: {chunks[2]}")
        self.assertTrue(chunks[3].startswith("CV: Candidate B"),
                        f"Chunk 3 should start with 'CV: Candidate B': {chunks[3]}")
        self.assertFalse(chunks[4].startswith("CV:"), f"Chunk 4 should not start with 'CV:', got: {chunks[4]}")
        self.assertFalse(chunks[5].startswith("CV:"), f"Chunk 5 should not start with 'CV:', got: {chunks[5]}")

    def test_default_max_tokens(self):
        # Using the default max_tokens (300) should merge most paragraphs,
        # resulting in two chunks (one per candidate).
        chunks = chunk_text(self.sample_text)  # uses default max_tokens=300
        self.assertEqual(len(chunks), 2, f"Expected 2 chunks for default max_tokens, got {len(chunks)}: {chunks}")
        self.assertTrue(chunks[0].startswith("CV: Candidate A"),
                        f"Chunk 0 should start with 'CV: Candidate A': {chunks[0]}")
        self.assertTrue(chunks[1].startswith("CV: Candidate B"),
                        f"Chunk 1 should start with 'CV: Candidate B': {chunks[1]}")

if __name__ == "__main__":
    unittest.main(exit=False)