"""Tests for cs336_basics.training.utils module."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cs336_basics.training import (
    calculate_params,
    estimate_checkpoint_size_mb,
    load_tokens,
    setup_device,
)


# =============================================================================
# Test calculate_params()
# =============================================================================

class TestCalculateParams:
    """Tests for parameter count calculation."""

    def test_minimal_model(self):
        """Test with a minimal model configuration."""
        settings = {
            "vocab_size": 100,
            "d_model": 64,
            "num_layers": 1,
            "d_ff": 128,
        }
        params = calculate_params(settings)
        
        # Manual calculation:
        # embeddings: 2 * 100 * 64 = 12,800
        # final norm: 64
        # per layer:
        #   norms: 2 * 64 = 128
        #   attention (Q,K,V,O): 4 * 64 * 64 = 16,384
        #   FFN (up, gate, down): 3 * 64 * 128 = 24,576
        # Total: 12,800 + 64 + 128 + 16,384 + 24,576 = 53,952
        expected = 12_800 + 64 + (128 + 16_384 + 24_576)
        assert params == expected

    def test_assignment_default_model(self):
        """Test the assignment default model (~22.7M params)."""
        settings = {
            "vocab_size": 10000,
            "d_model": 512,
            "num_layers": 4,
            "d_ff": 1344,
        }
        params = calculate_params(settings)
        
        # Verify exact calculation
        expected = (
            2 * 10000 * 512 +  # embeddings: 10,240,000
            512 +  # final norm: 512
            4 * (  # 4 layers
                2 * 512 +  # norms: 1,024
                4 * 512 * 512 +  # attention: 1,048,576
                3 * 512 * 1344  # FFN: 2,064,384
            )
        )
        assert params == expected
        
        # Should be approximately 22.7M parameters
        assert 22_000_000 < params < 23_000_000

    def test_scaling_with_layers(self):
        """Test that params scale linearly with num_layers."""
        base_settings = {
            "vocab_size": 1000,
            "d_model": 256,
            "num_layers": 1,
            "d_ff": 512,
        }
        
        params_1_layer = calculate_params(base_settings)
        
        base_settings["num_layers"] = 2
        params_2_layers = calculate_params(base_settings)
        
        base_settings["num_layers"] = 4
        params_4_layers = calculate_params(base_settings)
        
        # Per-layer params should be constant
        per_layer = params_2_layers - params_1_layer
        assert params_4_layers - params_2_layers == 2 * per_layer

    def test_scaling_with_d_model(self):
        """Test that attention params scale quadratically with d_model."""
        settings_small = {
            "vocab_size": 100,
            "d_model": 64,
            "num_layers": 1,
            "d_ff": 64,  # Keep FFN small to isolate attention
        }
        settings_large = {
            "vocab_size": 100,
            "d_model": 128,
            "num_layers": 1,
            "d_ff": 64,
        }
        
        params_small = calculate_params(settings_small)
        params_large = calculate_params(settings_large)
        
        # Attention params (4 * d_model^2) should quadruple when d_model doubles
        # But embeddings and FFN also change, so just verify it's more than linear
        assert params_large > 2 * params_small


# =============================================================================
# Test estimate_checkpoint_size_mb()
# =============================================================================

class TestEstimateCheckpointSize:
    """Tests for checkpoint size estimation."""

    def test_basic_calculation(self):
        """Test that checkpoint size is 12 bytes per parameter."""
        settings = {
            "vocab_size": 1000,
            "d_model": 64,
            "num_layers": 1,
            "d_ff": 128,
        }
        
        num_params = calculate_params(settings)
        size_mb = estimate_checkpoint_size_mb(settings)
        
        # 12 bytes per param (4 for model + 4 for m + 4 for v)
        expected_mb = num_params * 12 / (1024 * 1024)
        assert abs(size_mb - expected_mb) < 0.001

    def test_assignment_default_size(self):
        """Test checkpoint size for assignment default model."""
        settings = {
            "vocab_size": 10000,
            "d_model": 512,
            "num_layers": 4,
            "d_ff": 1344,
        }
        
        size_mb = estimate_checkpoint_size_mb(settings)
        
        # ~22.7M params * 12 bytes = ~272MB, should be around 250-280 MB
        assert 250 < size_mb < 280

    def test_proportional_to_params(self):
        """Test that checkpoint size is proportional to parameter count."""
        settings_small = {
            "vocab_size": 1000,
            "d_model": 64,
            "num_layers": 1,
            "d_ff": 128,
        }
        settings_large = {
            "vocab_size": 1000,
            "d_model": 64,
            "num_layers": 2,
            "d_ff": 128,
        }
        
        size_small = estimate_checkpoint_size_mb(settings_small)
        size_large = estimate_checkpoint_size_mb(settings_large)
        params_small = calculate_params(settings_small)
        params_large = calculate_params(settings_large)
        
        # Ratio should be the same
        assert abs(size_large / size_small - params_large / params_small) < 0.001


# =============================================================================
# Test load_tokens()
# =============================================================================

class TestLoadTokens:
    """Tests for token loading functionality."""

    def test_load_valid_files(self):
        """Test loading valid numpy files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.npy")
            valid_path = os.path.join(tmpdir, "valid.npy")
            
            # Create test data
            train_data = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
            valid_data = np.array([6, 7, 8], dtype=np.uint16)
            np.save(train_path, train_data)
            np.save(valid_path, valid_data)
            
            # Load and verify
            train_tokens, valid_tokens = load_tokens(train_path, valid_path, verbose=False)
            
            assert len(train_tokens) == 5
            assert len(valid_tokens) == 3
            np.testing.assert_array_equal(train_tokens, train_data)
            np.testing.assert_array_equal(valid_tokens, valid_data)

    def test_missing_train_file(self):
        """Test error when training file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_path = os.path.join(tmpdir, "valid.npy")
            np.save(valid_path, np.array([1, 2, 3]))
            
            with pytest.raises(FileNotFoundError, match="Training data not found"):
                load_tokens("/nonexistent/train.npy", valid_path, verbose=False)

    def test_missing_valid_file(self):
        """Test error when validation file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.npy")
            np.save(train_path, np.array([1, 2, 3]))
            
            with pytest.raises(FileNotFoundError, match="Validation data not found"):
                load_tokens(train_path, "/nonexistent/valid.npy", verbose=False)

    def test_memory_mapped(self):
        """Test that loaded arrays are memory-mapped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.npy")
            valid_path = os.path.join(tmpdir, "valid.npy")
            
            # Create larger test data
            train_data = np.arange(10000, dtype=np.uint16)
            valid_data = np.arange(1000, dtype=np.uint16)
            np.save(train_path, train_data)
            np.save(valid_path, valid_data)
            
            train_tokens, valid_tokens = load_tokens(train_path, valid_path, verbose=False)
            
            # Memory-mapped arrays have a specific type
            assert isinstance(train_tokens, np.memmap)
            assert isinstance(valid_tokens, np.memmap)


# =============================================================================
# Test setup_device()
# =============================================================================

class TestSetupDevice:
    """Tests for device setup and seeding."""

    def test_returns_valid_device(self):
        """Test that setup_device returns a valid device string."""
        device = setup_device(seed=42, verbose=False)
        assert device in ("cuda", "mps", "cpu")

    def test_seeds_numpy(self):
        """Test that numpy random seed is set."""
        setup_device(seed=12345, verbose=False)
        val1 = np.random.rand()
        
        setup_device(seed=12345, verbose=False)
        val2 = np.random.rand()
        
        assert val1 == val2

    def test_seeds_torch(self):
        """Test that torch random seed is set."""
        setup_device(seed=12345, verbose=False)
        val1 = torch.rand(1).item()
        
        setup_device(seed=12345, verbose=False)
        val2 = torch.rand(1).item()
        
        assert val1 == val2

    def test_different_seeds_different_values(self):
        """Test that different seeds produce different random values."""
        setup_device(seed=111, verbose=False)
        val1 = np.random.rand()
        
        setup_device(seed=222, verbose=False)
        val2 = np.random.rand()
        
        assert val1 != val2


# =============================================================================
# Test model config validation
# =============================================================================

class TestModelConfigValidation:
    """Tests for model configuration validation."""

    @pytest.fixture
    def model_configs(self):
        """Load model configs from YAML."""
        config_path = Path(__file__).parent.parent / "configs" / "models.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_all_configs_have_required_keys(self, model_configs):
        """Test that all model configs have required top-level keys."""
        required_keys = {"name", "batch_size", "model_settings"}
        
        for model_name, config in model_configs.items():
            missing = required_keys - set(config.keys())
            assert not missing, f"{model_name} missing keys: {missing}"

    def test_all_configs_have_required_model_settings(self, model_configs):
        """Test that all model configs have required model_settings keys."""
        required_settings = {
            "vocab_size", "d_model", "num_heads", "num_layers",
            "d_ff", "context_length", "rope_theta"
        }
        
        for model_name, config in model_configs.items():
            settings = config["model_settings"]
            missing = required_settings - set(settings.keys())
            assert not missing, f"{model_name} missing model_settings: {missing}"

    def test_d_model_divisible_by_num_heads(self, model_configs):
        """Test that d_model is divisible by num_heads for all models."""
        for model_name, config in model_configs.items():
            settings = config["model_settings"]
            d_model = settings["d_model"]
            num_heads = settings["num_heads"]
            assert d_model % num_heads == 0, (
                f"{model_name}: d_model ({d_model}) not divisible by num_heads ({num_heads})"
            )

    def test_positive_values(self, model_configs):
        """Test that all numeric values are positive."""
        for model_name, config in model_configs.items():
            assert config["batch_size"] > 0, f"{model_name}: batch_size must be positive"
            
            settings = config["model_settings"]
            for key in ["vocab_size", "d_model", "num_heads", "num_layers", "d_ff", "context_length"]:
                assert settings[key] > 0, f"{model_name}: {key} must be positive"

    def test_expected_models_present(self, model_configs):
        """Test that expected model configurations are present."""
        expected = {"model_a", "model_b", "smol_model_a", "smol_model_b", "assignment_default"}
        actual = set(model_configs.keys())
        assert expected == actual, f"Expected {expected}, got {actual}"

    def test_smol_models_smaller_than_full(self, model_configs):
        """Test that smol models have fewer parameters than full models."""
        params_a = calculate_params(model_configs["model_a"]["model_settings"])
        params_smol_a = calculate_params(model_configs["smol_model_a"]["model_settings"])
        
        params_b = calculate_params(model_configs["model_b"]["model_settings"])
        params_smol_b = calculate_params(model_configs["smol_model_b"]["model_settings"])
        
        assert params_smol_a < params_a, "smol_model_a should be smaller than model_a"
        assert params_smol_b < params_b, "smol_model_b should be smaller than model_b"
