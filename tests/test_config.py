"""
Tests for configuration loading and validation.
"""

import pytest
import yaml
import tempfile
from pathlib import Path


def test_load_base_config():
    """Test that base config loads successfully."""
    config_path = Path("train/configs/base.yaml")
    
    if not config_path.exists():
        pytest.skip("base.yaml not found")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check required sections
    assert "environment" in config
    assert "ppo" in config
    assert "training" in config
    
    # Check environment params
    assert "grid_size" in config["environment"]
    assert config["environment"]["grid_size"] > 0
    
    # Check PPO params
    assert "learning_rate" in config["ppo"]
    assert config["ppo"]["learning_rate"] > 0
    
    # Check training params
    assert "total_timesteps" in config["training"]
    assert config["training"]["total_timesteps"] > 0


def test_config_override():
    """Test that config values can be overridden."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            "environment": {"grid_size": 10},
            "ppo": {"learning_rate": 0.001},
            "training": {"total_timesteps": 1000},
        }, f)
        temp_config = f.name
    
    try:
        with open(temp_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override
        config["environment"]["grid_size"] = 15
        config["training"]["total_timesteps"] = 2000
        
        assert config["environment"]["grid_size"] == 15
        assert config["training"]["total_timesteps"] == 2000
        assert config["ppo"]["learning_rate"] == 0.001  # Unchanged
    finally:
        Path(temp_config).unlink()


def test_all_configs_valid():
    """Test that all provided configs are valid YAML."""
    config_dir = Path("train/configs")
    
    if not config_dir.exists():
        pytest.skip("configs directory not found")
    
    config_files = list(config_dir.glob("**/*.yaml"))
    
    assert len(config_files) > 0, "No config files found"
    
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config, dict), f"Invalid config in {config_file}"
        print(f"✓ {config_file.name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])