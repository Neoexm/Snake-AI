"""
Real-time resource monitoring script.

Displays live CPU, RAM, and GPU utilization during training.
"""

import sys
import time
import argparse
from pathlib import Path

import psutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def format_bytes(bytes_val):
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def get_gpu_stats():
    """Get GPU statistics if available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        stats = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_alloc = torch.cuda.memory_allocated(i)
            mem_reserved = torch.cuda.memory_reserved(i)
            mem_total = props.total_memory
            
            stats.append({
                'id': i,
                'name': props.name,
                'memory_allocated': mem_alloc,
                'memory_reserved': mem_reserved,
                'memory_total': mem_total,
                'memory_percent': (mem_alloc / mem_total) * 100,
            })
        
        return stats
    except ImportError:
        return None
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return None


def print_stats(interval: float = 1.0, compact: bool = False):
    """
    Print resource statistics at regular intervals.
    
    Parameters
    ----------
    interval : float
        Update interval in seconds.
    compact : bool
        Use compact single-line output.
    """
    try:
        iteration = 0
        while True:
            iteration += 1
            
            # CPU stats
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # RAM stats
            mem = psutil.virtual_memory()
            
            # GPU stats
            gpu_stats = get_gpu_stats()
            
            if compact:
                # Single line output
                output = f"[{iteration}] CPU: {cpu_percent:5.1f}% | RAM: {mem.percent:5.1f}% ({format_bytes(mem.used)}/{format_bytes(mem.total)})"
                
                if gpu_stats:
                    for gpu in gpu_stats:
                        output += f" | GPU{gpu['id']}: {gpu['memory_percent']:5.1f}% ({format_bytes(gpu['memory_allocated'])})"
                
                print(f"\r{output}", end='', flush=True)
            else:
                # Multi-line detailed output
                print("\n" + "="*60)
                print(f"Resource Monitor - Iteration {iteration}")
                print("="*60)
                
                print(f"\n📊 CPU")
                print(f"  Usage: {cpu_percent:.1f}%")
                print(f"  Cores: {cpu_count} (physical: {psutil.cpu_count(logical=False)})")
                
                print(f"\n💾 RAM")
                print(f"  Usage: {mem.percent:.1f}%")
                print(f"  Used: {format_bytes(mem.used)} / {format_bytes(mem.total)}")
                print(f"  Available: {format_bytes(mem.available)}")
                
                if gpu_stats:
                    print(f"\n🎮 GPU")
                    for gpu in gpu_stats:
                        print(f"\n  Device {gpu['id']}: {gpu['name']}")
                        print(f"    Memory Usage: {gpu['memory_percent']:.1f}%")
                        print(f"    Allocated: {format_bytes(gpu['memory_allocated'])}")
                        print(f"    Reserved: {format_bytes(gpu['memory_reserved'])}")
                        print(f"    Total: {format_bytes(gpu['memory_total'])}")
                else:
                    print(f"\n🎮 GPU: Not available")
                
                print("\n" + "="*60)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        if compact:
            print()  # New line after compact output
        print("\n👋 Monitoring stopped")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Monitor system resources in real-time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Update interval in seconds",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use compact single-line output",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("🔍 Starting resource monitoring...")
    print(f"Update interval: {args.interval}s")
    print(f"Mode: {'Compact' if args.compact else 'Detailed'}")
    print("\nPress Ctrl+C to stop\n")
    
    time.sleep(1)
    print_stats(interval=args.interval, compact=args.compact)


if __name__ == "__main__":
    main()