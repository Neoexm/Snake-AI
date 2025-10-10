import os
import sys
import argparse
import yaml
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from supervised.utils import (
    set_seed, get_device, autoscale_workers,
    save_json, TimeManager, AverageMeter
)
from supervised.dataset import create_dataloaders
from supervised.models import make_bc_model
from snake_env import SnakeEnv
import gymnasium as gym


def train_epoch(
    model, train_loader, criterion, optimizer, scaler, device, use_amp, grad_accum_steps, epoch_num=0
):
    """Train for one epoch with gradient accumulation."""
    model.train()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    grad_norm_meter = AverageMeter()
    
    optimizer.zero_grad()
    
    start_time = time.time()
    
    pbar = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch_num}",
        unit="batch",
        ncols=120,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    try:
        for batch_idx, (obs, actions) in enumerate(pbar):
            obs = obs.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(obs)
                    loss = criterion(logits, actions) / grad_accum_steps
            else:
                logits = model(obs)
                loss = criterion(logits, actions) / grad_accum_steps
            
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                grad_norm_meter.update(grad_norm.item(), 1)
            
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == actions).float().mean()
                acc_val = acc.item()
                loss_val = loss.item()
            
            loss_meter.update(loss_val * grad_accum_steps, obs.size(0))
            acc_meter.update(acc_val, obs.size(0))
            
            del obs, actions, logits, loss, preds, acc
            
            if (batch_idx + 1) % 50 == 0:
                import gc
                gc.collect()
            
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.3f}',
                'grad': f'{grad_norm_meter.avg:.2f}' if grad_norm_meter.count > 0 else 'N/A'
            })
    except Exception as e:
        pbar.close()
        print(f"\nERROR during training epoch: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    pbar.close()
    return loss_meter.avg, acc_meter.avg, grad_norm_meter.avg


def validate(model, val_loader, criterion, device, use_amp):
    """Validate model."""
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(
        val_loader,
        desc="Validation",
        unit="batch",
        ncols=120,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )
    
    with torch.no_grad():
        for obs, actions in pbar:
            obs = obs.to(device)
            actions = actions.to(device)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(obs)
                    loss = criterion(logits, actions)
            else:
                logits = model(obs)
                loss = criterion(logits, actions)
            
            preds = logits.argmax(dim=1)
            acc = (preds == actions).float().mean()
            
            loss_meter.update(loss.item(), obs.size(0))
            acc_meter.update(acc.item(), obs.size(0))
            
            del obs, actions, logits, loss, preds, acc
            
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.3f}'
            })
    
    pbar.close()
    return loss_meter.avg, acc_meter.avg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train behavior cloning policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--config', type=str, required=True, help='Path to BC config')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--time-budget-min', type=int, default=None, help='Time budget in minutes')
    parser.add_argument('--save-dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', help='Force deterministic training')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size from config')
    parser.add_argument('--grad-accum', type=int, default=None, help='Override grad accum steps from config')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs from config')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set PyTorch to use minimal memory
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Disable unnecessary caching
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Force garbage collection
    import gc
    gc.collect()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.grad_accum:
        config['grad_accum_steps'] = args.grad_accum
    if args.epochs:
        config['epochs'] = args.epochs
    
    device = get_device(args.device)
    deterministic = args.deterministic or config.get('deterministic', True)
    set_seed(args.seed, deterministic=deterministic)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("BEHAVIOR CLONING TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Data: {args.data}")
    print(f"Save dir: {save_dir}")
    if args.time_budget_min:
        print(f"Time budget: {args.time_budget_min} minutes")
    print("="*60 + "\n")
    
    time_manager = TimeManager(args.time_budget_min)
    
    num_workers = config.get('num_workers', 'auto')
    if num_workers == 'auto':
        num_workers = autoscale_workers() if sys.platform != 'win32' else 0
    
    pin_memory = config.get('pin_memory', 'auto')
    if pin_memory == 'auto':
        pin_memory = device == 'cuda'
    
    use_amp = config.get('amp', 'auto')
    if use_amp == 'auto':
        use_amp = device == 'cuda'
    elif use_amp == 'true':
        use_amp = True
    else:
        use_amp = False
    
    grad_accum_steps = config.get('grad_accum_steps', 1)
    effective_batch_size = config['batch_size'] * grad_accum_steps
    
    print("Loading dataset...")
    print("Using F:/snake_bc_cache for temporary storage...")
    train_loader, val_loader, meta = create_dataloaders(
        data_dir=args.data,
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        num_workers=num_workers,
        pin_memory=pin_memory,
        augment=config.get('augment', {}).get('enabled', False),
        cache_dir="F:/snake_bc_cache"
    )
    
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Grad accum steps: {grad_accum_steps}")
    print(f"  EFFECTIVE batch size: {effective_batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Pin memory: {pin_memory}")
    print(f"  Mixed precision: {use_amp}\n")
    
    obs_shape = tuple(meta['obs_shape'])
    action_dim = meta['action_space']
    
    obs_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype='float32')
    action_space = gym.spaces.Discrete(action_dim)
    
    print("Creating model...")
    model = make_bc_model(obs_space, action_space, config)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    print(f"  Obs shape: {obs_shape}")
    print(f"  Action dim: {action_dim}\n")
    
    criterion = nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    writer = SummaryWriter(log_dir=str(save_dir))
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...\n")
    
    for epoch in range(config['epochs']):
        if time_manager.is_expired():
            print(f"\nTime budget expired ({args.time_budget_min} min)")
            break
        
        train_loss, train_acc, grad_norm = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, grad_accum_steps, epoch_num=epoch+1
        )
        
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, use_amp
        )
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Grad/norm', grad_norm, epoch)
        writer.add_scalar('Time/elapsed_min', time_manager.elapsed_minutes(), epoch)
        
        elapsed_min = int(time_manager.elapsed_minutes())
        elapsed_sec = int((time_manager.elapsed_minutes() - elapsed_min) * 60)
        print(f"Epoch {epoch+1}/{config['epochs']} Summary: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
              f"grad_norm={grad_norm:.3f} | "
              f"time={elapsed_min}m{elapsed_sec:02d}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config,
                'meta': meta
            }, save_dir / 'best.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= config.get('early_stop_patience', 10):
            print(f"\nEarly stopping triggered (patience={config['early_stop_patience']})")
            break
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'meta': meta
    }, save_dir / 'last.pt')
    
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    metrics = {
        'final_train_loss': train_loss,
        'final_train_acc': train_acc,
        'best_val_loss': best_val_loss,
        'final_val_loss': val_loss,
        'final_val_acc': val_acc,
        'total_epochs': epoch + 1,
        'elapsed_minutes': time_manager.elapsed_minutes()
    }
    
    save_json(metrics, save_dir / 'metrics.json')
    save_json({'num_classes': action_dim}, save_dir / 'classes.json')
    
    writer.close()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Final val acc: {val_acc:.4f}")
    print(f"Total epochs: {epoch + 1}")
    print(f"Elapsed time: {time_manager.elapsed_minutes():.1f} minutes")
    print(f"Models saved to: {save_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
