"""Custom tests for tokenizer save/load functionality."""

import os
import tempfile
import pytest
from cs336_basics.bpe import train_bpe, save_bpe, Tokenizer


def load_bpe(input_path: str | os.PathLike) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Load vocab and merges from directory (matching save_bpe format)."""
    import json
    
    vocab_path = os.path.join(input_path, "vocab.json")
    merges_path = os.path.join(input_path, "merges.txt")
    
    # Load vocab - latin-1 encoded
    with open(vocab_path, "r", encoding="latin-1") as f:
        vocab_str = json.load(f)
    vocab = {int(k): v.encode('latin-1') for k, v in vocab_str.items()}
    
    # Load merges - tab-separated, latin-1 encoded
    merges = []
    with open(merges_path, "r", encoding="latin-1") as f:
        for line in f:
            a, b = line.rstrip('\n').split('\t')
            merges.append((a.encode('latin-1'), b.encode('latin-1')))
    
    return vocab, merges


class TestSaveLoadBPE:
    """Tests for save_bpe and load_bpe roundtrip."""
    
    def test_save_load_roundtrip_small_corpus(self):
        """Test that save/load preserves vocab and merges exactly."""
        # Use the small test corpus
        corpus_path = "tests/fixtures/corpus.en"
        vocab_size = 300  # Small for fast test
        special_tokens = ["<|endoftext|>"]
        
        # Train
        vocab, merges = train_bpe(corpus_path, vocab_size, special_tokens)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            save_bpe(tmpdir, vocab, merges)
            
            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, "vocab.json"))
            assert os.path.exists(os.path.join(tmpdir, "merges.txt"))
            
            # Load
            loaded_vocab, loaded_merges = load_bpe(tmpdir)
            
            # Compare vocab
            assert len(loaded_vocab) == len(vocab), "Vocab size mismatch"
            for k, v in vocab.items():
                assert k in loaded_vocab, f"Missing vocab key {k}"
                assert loaded_vocab[k] == v, f"Vocab value mismatch for key {k}: {loaded_vocab[k]} != {v}"
            
            # Compare merges
            assert len(loaded_merges) == len(merges), "Merges count mismatch"
            for i, (orig, loaded) in enumerate(zip(merges, loaded_merges)):
                assert orig == loaded, f"Merge mismatch at index {i}: {orig} != {loaded}"
    
    def test_tokenizer_from_saved_files(self):
        """Test that a Tokenizer built from saved files encodes identically."""
        corpus_path = "tests/fixtures/corpus.en"
        vocab_size = 300
        special_tokens = ["<|endoftext|>"]
        
        # Train and create original tokenizer
        vocab, merges = train_bpe(corpus_path, vocab_size, special_tokens)
        original_tokenizer = Tokenizer(vocab, merges, special_tokens)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save and reload
            save_bpe(tmpdir, vocab, merges)
            loaded_vocab, loaded_merges = load_bpe(tmpdir)
            loaded_tokenizer = Tokenizer(loaded_vocab, loaded_merges, special_tokens)
            
            # Test encoding matches
            test_texts = [
                "Hello, world!",
                "The quick brown fox jumps over the lazy dog.",
                "Testing 123... <|endoftext|> More text here.",
                "Unicode: café, naïve, 日本語",
            ]
            
            for text in test_texts:
                original_ids = original_tokenizer.encode(text)
                loaded_ids = loaded_tokenizer.encode(text)
                assert original_ids == loaded_ids, f"Encoding mismatch for: {text}"
    
    def test_roundtrip_preserves_all_256_base_bytes(self):
        """Ensure all 256 base vocab bytes survive save/load."""
        # Create minimal vocab with all 256 bytes
        vocab = {i: bytes([i]) for i in range(256)}
        merges = []  # No merges needed for this test
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_bpe(tmpdir, vocab, merges)
            loaded_vocab, loaded_merges = load_bpe(tmpdir)
            
            # Check all 256 bytes are preserved
            for i in range(256):
                assert i in loaded_vocab, f"Missing byte {i}"
                assert loaded_vocab[i] == bytes([i]), f"Byte {i} corrupted: {loaded_vocab[i]}"
    
    def test_merges_with_special_bytes(self):
        """Test merges containing bytes that are invalid UTF-8."""
        vocab = {i: bytes([i]) for i in range(256)}
        # Add a merge with bytes 0x80 and 0x81 (invalid standalone UTF-8)
        vocab[256] = bytes([0x80, 0x81])
        merges = [(bytes([0x80]), bytes([0x81]))]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_bpe(tmpdir, vocab, merges)
            loaded_vocab, loaded_merges = load_bpe(tmpdir)
            
            assert loaded_merges == merges, "Merge with special bytes corrupted"
            assert loaded_vocab[256] == bytes([0x80, 0x81]), "Merged token corrupted"


class TestSaveLoadFilesExist:
    """Test checking for existing tokenizer files."""
    
    def test_check_existing_files(self):
        """Test pattern for checking if tokenizer already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = os.path.join(tmpdir, "vocab.json")
            merges_path = os.path.join(tmpdir, "merges.txt")
            
            # Initially doesn't exist
            assert not (os.path.exists(vocab_path) and os.path.exists(merges_path))
            
            # Create minimal files
            vocab = {i: bytes([i]) for i in range(256)}
            merges = []
            save_bpe(tmpdir, vocab, merges)
            
            # Now exists
            assert os.path.exists(vocab_path) and os.path.exists(merges_path)

