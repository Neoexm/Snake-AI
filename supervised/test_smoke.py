import sys
import tempfile
import shutil
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from snake_env import SnakeEnv, RewardShaping, FrameStack
from supervised.models import make_bc_model
from supervised.dataset import SnakeBCDataset
import gymnasium as gym


def test_env_creation():
    """Test environment creation."""
    print("Testing environment creation...")
    env = SnakeEnv(grid_size=12)
    env = RewardShaping(env)
    env = FrameStack(env, n_frames=2)
    
    obs, _ = env.reset(seed=0)
    assert obs.shape == (6, 12, 12), f"Expected (6, 12, 12), got {obs.shape}"
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.close()
    print("  ✓ Environment test passed")


def test_model_creation():
    """Test model creation for CNN and MLP."""
    print("Testing model creation...")
    
    config = {'dropout': 0.1, 'cnn_width': 32}
    
    obs_space_cnn = gym.spaces.Box(low=0, high=1, shape=(6, 12, 12), dtype='float32')
    action_space = gym.spaces.Discrete(4)
    
    model = make_bc_model(obs_space_cnn, action_space, config)
    
    x = torch.randn(4, 6, 12, 12)
    logits = model(x)
    assert logits.shape == (4, 4), f"Expected (4, 4), got {logits.shape}"
    
    print("  ✓ Model test passed")


def test_dataset():
    """Test dataset creation and loading."""
    print("Testing dataset...")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        observations = [np.random.rand(3, 12, 12).astype(np.float32) for _ in range(64)]
        actions = [np.random.randint(0, 4) for _ in range(64)]
        
        shard_data = {
            'observations': torch.from_numpy(np.stack(observations)),
            'actions': torch.tensor(actions, dtype=torch.long)
        }
        torch.save(shard_data, temp_dir / 'shard_00001.pt')
        
        meta = {
            'total_steps': 64,
            'obs_shape': [3, 12, 12],
            'action_space': 4,
            'allow_hflip': False
        }
        
        import json
        with open(temp_dir / 'meta.json', 'w') as f:
            json.dump(meta, f)
        
        dataset = SnakeBCDataset(str(temp_dir))
        assert len(dataset) == 64
        
        obs, action = dataset[0]
        assert obs.shape == (3, 12, 12)
        assert 0 <= action < 4
        
        print("  ✓ Dataset test passed")
        
    finally:
        shutil.rmtree(temp_dir)


def test_forward_pass():
    """Test full forward pass through model."""
    print("Testing forward pass...")
    
    config = {'dropout': 0.1, 'cnn_width': 32}
    obs_space = gym.spaces.Box(low=0, high=1, shape=(3, 12, 12), dtype='float32')
    action_space = gym.spaces.Discrete(4)
    
    model = make_bc_model(obs_space, action_space, config)
    model.eval()
    
    obs = torch.rand(1, 3, 12, 12)
    
    with torch.no_grad():
        logits = model(obs)
        action = logits.argmax(dim=1).item()
    
    assert 0 <= action < 4, f"Invalid action: {action}"
    
    print("  ✓ Forward pass test passed")


def test_training_loop():
    """Test minimal training loop."""
    print("Testing training loop (2 epochs)...")
    
    import tempfile
    import shutil
    from supervised.utils import set_seed
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import torch.nn as nn
    
    set_seed(42)
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        observations = [np.random.rand(3, 12, 12).astype(np.float32) for _ in range(128)]
        actions = [np.random.randint(0, 4) for _ in range(128)]
        
        shard_data = {
            'observations': torch.from_numpy(np.stack(observations)),
            'actions': torch.tensor(actions, dtype=torch.long)
        }
        torch.save(shard_data, temp_dir / 'shard_00001.pt')
        
        meta = {
            'total_steps': 128,
            'obs_shape': [3, 12, 12],
            'action_space': 4,
            'allow_hflip': False
        }
        
        import json
        with open(temp_dir / 'meta.json', 'w') as f:
            json.dump(meta, f)
        
        from supervised.dataset import SnakeBCDataset
        dataset = SnakeBCDataset(str(temp_dir))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        config = {'dropout': 0.1, 'cnn_width': 32}
        obs_space = gym.spaces.Box(low=0, high=1, shape=(3, 12, 12), dtype='float32')
        action_space = gym.spaces.Discrete(4)
        
        model = make_bc_model(obs_space, action_space, config)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(2):
            total_loss = 0
            for obs, acts in loader:
                optimizer.zero_grad()
                logits = model(obs)
                loss = criterion(logits, acts)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"    Epoch {epoch+1}/2: loss={total_loss/len(loader):.4f}")
        
        print("  ✓ Training loop test passed")
        
    finally:
        shutil.rmtree(temp_dir)


def main():
    print("="*60)
    print("SUPERVISED BC SMOKE TEST")
    print("="*60 + "\n")
    
    test_env_creation()
    test_model_creation()
    test_dataset()
    test_forward_pass()
    test_training_loop()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)


if __name__ == '__main__':
    main()
