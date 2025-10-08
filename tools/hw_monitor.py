"""
Hardware monitoring for GPU/CPU/RAM utilization during training.

This module provides real-time monitoring of system resources using NVML for GPU stats
and psutil for CPU/RAM stats. Designed to be used as a background thread during training.
"""

import time
import threading
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import warnings

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    warnings.warn("pynvml not available. Install with: pip install nvidia-ml-py3")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Install with: pip install psutil")


@dataclass
class GPUStats:
    """Statistics for a single GPU."""
    device_id: int
    name: str
    utilization_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    temperature_c: Optional[float]
    power_draw_w: Optional[float]
    power_limit_w: Optional[float]


@dataclass
class SystemStats:
    """Complete system statistics snapshot."""
    timestamp: float
    gpus: List[GPUStats]
    cpu_percent: float
    cpu_count: int
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float


class HardwareMonitor:
    """
    Monitor hardware utilization in a background thread.
    
    Polls GPU (via NVML), CPU, and RAM stats at a specified interval and optionally
    logs to CSV and TensorBoard.
    
    Parameters
    ----------
    poll_interval : float
        Seconds between polls (default: 1.0)
    log_dir : Path or str, optional
        Directory to save CSV logs
    tensorboard_logger : optional
        SB3 logger instance for TensorBoard integration
    """
    
    def __init__(
        self,
        poll_interval: float = 1.0,
        log_dir: Optional[Path] = None,
        tensorboard_logger=None,
    ):
        self.poll_interval = poll_interval
        self.log_dir = Path(log_dir) if log_dir else None
        self.tensorboard_logger = tensorboard_logger
        
        self._running = False
        self._thread = None
        self._stats_history: List[SystemStats] = []
        self._nvml_initialized = False
        
        # Initialize NVML if available
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
                self.gpu_count = pynvml.nvmlDeviceGetCount()
            except Exception as e:
                warnings.warn(f"Failed to initialize NVML: {e}")
                self.gpu_count = 0
        else:
            self.gpu_count = 0
        
        # Open CSV log if log_dir provided
        self.csv_file = None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.csv_file = open(self.log_dir / "hardware_stats.csv", "w")
            self._write_csv_header()
    
    def _write_csv_header(self):
        """Write CSV header."""
        if not self.csv_file:
            return
        
        header = ["timestamp", "cpu_percent", "ram_percent", "ram_used_gb"]
        for i in range(self.gpu_count):
            header.extend([
                f"gpu{i}_util_percent",
                f"gpu{i}_mem_percent",
                f"gpu{i}_mem_used_gb",
                f"gpu{i}_temp_c",
                f"gpu{i}_power_w",
            ])
        self.csv_file.write(",".join(header) + "\n")
        self.csv_file.flush()
    
    def _get_gpu_stats(self) -> List[GPUStats]:
        """Get stats for all GPUs."""
        if not self._nvml_initialized:
            return []
        
        stats = []
        for i in range(self.gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                
                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used_gb = mem_info.used / (1024**3)
                mem_total_gb = mem_info.total / (1024**3)
                mem_percent = (mem_info.used / mem_info.total) * 100
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = None
                
                # Power
                try:
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                except:
                    power_draw = None
                    power_limit = None
                
                # Name
                name = pynvml.nvmlDeviceGetName(handle)
                
                stats.append(GPUStats(
                    device_id=i,
                    name=name,
                    utilization_percent=gpu_util,
                    memory_used_gb=mem_used_gb,
                    memory_total_gb=mem_total_gb,
                    memory_percent=mem_percent,
                    temperature_c=temp,
                    power_draw_w=power_draw,
                    power_limit_w=power_limit,
                ))
            except Exception as e:
                warnings.warn(f"Error reading GPU {i}: {e}")
        
        return stats
    
    def _get_cpu_stats(self) -> Dict[str, Any]:
        """Get CPU stats."""
        if not PSUTIL_AVAILABLE:
            return {"cpu_percent": 0.0, "cpu_count": 0}
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(),
        }
    
    def _get_ram_stats(self) -> Dict[str, Any]:
        """Get RAM stats."""
        if not PSUTIL_AVAILABLE:
            return {"ram_used_gb": 0.0, "ram_total_gb": 0.0, "ram_percent": 0.0}
        
        mem = psutil.virtual_memory()
        return {
            "ram_used_gb": mem.used / (1024**3),
            "ram_total_gb": mem.total / (1024**3),
            "ram_percent": mem.percent,
        }
    
    def get_current_stats(self) -> SystemStats:
        """Get current system stats snapshot."""
        gpu_stats = self._get_gpu_stats()
        cpu_stats = self._get_cpu_stats()
        ram_stats = self._get_ram_stats()
        
        return SystemStats(
            timestamp=time.time(),
            gpus=gpu_stats,
            **cpu_stats,
            **ram_stats,
        )
    
    def _monitor_loop(self):
        """Main monitoring loop (runs in thread)."""
        while self._running:
            stats = self.get_current_stats()
            self._stats_history.append(stats)
            
            # Log to CSV
            if self.csv_file:
                row = [
                    f"{stats.timestamp:.2f}",
                    f"{stats.cpu_percent:.1f}",
                    f"{stats.ram_percent:.1f}",
                    f"{stats.ram_used_gb:.2f}",
                ]
                for gpu in stats.gpus:
                    row.extend([
                        f"{gpu.utilization_percent:.1f}",
                        f"{gpu.memory_percent:.1f}",
                        f"{gpu.memory_used_gb:.2f}",
                        f"{gpu.temperature_c:.1f}" if gpu.temperature_c else "0.0",
                        f"{gpu.power_draw_w:.1f}" if gpu.power_draw_w else "0.0",
                    ])
                self.csv_file.write(",".join(row) + "\n")
                self.csv_file.flush()
            
            # Log to TensorBoard
            if self.tensorboard_logger:
                self.tensorboard_logger.record("system/cpu_percent", stats.cpu_percent)
                self.tensorboard_logger.record("system/ram_percent", stats.ram_percent)
                self.tensorboard_logger.record("system/ram_used_gb", stats.ram_used_gb)
                
                for gpu in stats.gpus:
                    prefix = f"system/gpu{gpu.device_id}"
                    self.tensorboard_logger.record(f"{prefix}_util_percent", gpu.utilization_percent)
                    self.tensorboard_logger.record(f"{prefix}_mem_percent", gpu.memory_percent)
                    self.tensorboard_logger.record(f"{prefix}_mem_used_gb", gpu.memory_used_gb)
                    if gpu.temperature_c is not None:
                        self.tensorboard_logger.record(f"{prefix}_temp_c", gpu.temperature_c)
                    if gpu.power_draw_w is not None:
                        self.tensorboard_logger.record(f"{prefix}_power_w", gpu.power_draw_w)
            
            time.sleep(self.poll_interval)
    
    def start(self):
        """Start monitoring in background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        
        if self.csv_file:
            self.csv_file.close()
    
    def get_average_stats(self, last_n_seconds: Optional[float] = None) -> Optional[SystemStats]:
        """
        Get average stats over last N seconds.
        
        Parameters
        ----------
        last_n_seconds : float, optional
            Time window in seconds (default: all history)
        
        Returns
        -------
        SystemStats or None
            Averaged stats, or None if no data
        """
        if not self._stats_history:
            return None
        
        if last_n_seconds is None:
            samples = self._stats_history
        else:
            cutoff_time = time.time() - last_n_seconds
            samples = [s for s in self._stats_history if s.timestamp >= cutoff_time]
        
        if not samples:
            return None
        
        # Average CPU/RAM
        avg_cpu = sum(s.cpu_percent for s in samples) / len(samples)
        avg_ram_percent = sum(s.ram_percent for s in samples) / len(samples)
        avg_ram_used = sum(s.ram_used_gb for s in samples) / len(samples)
        
        # Average GPU stats
        avg_gpus = []
        if samples[0].gpus:
            for gpu_id in range(len(samples[0].gpus)):
                gpu_utils = [s.gpus[gpu_id].utilization_percent for s in samples if len(s.gpus) > gpu_id]
                gpu_mems = [s.gpus[gpu_id].memory_percent for s in samples if len(s.gpus) > gpu_id]
                gpu_mem_gbs = [s.gpus[gpu_id].memory_used_gb for s in samples if len(s.gpus) > gpu_id]
                gpu_temps = [s.gpus[gpu_id].temperature_c for s in samples 
                            if len(s.gpus) > gpu_id and s.gpus[gpu_id].temperature_c is not None]
                gpu_powers = [s.gpus[gpu_id].power_draw_w for s in samples 
                             if len(s.gpus) > gpu_id and s.gpus[gpu_id].power_draw_w is not None]
                
                avg_gpus.append(GPUStats(
                    device_id=gpu_id,
                    name=samples[0].gpus[gpu_id].name,
                    utilization_percent=sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0,
                    memory_used_gb=sum(gpu_mem_gbs) / len(gpu_mem_gbs) if gpu_mem_gbs else 0.0,
                    memory_total_gb=samples[0].gpus[gpu_id].memory_total_gb,
                    memory_percent=sum(gpu_mems) / len(gpu_mems) if gpu_mems else 0.0,
                    temperature_c=sum(gpu_temps) / len(gpu_temps) if gpu_temps else None,
                    power_draw_w=sum(gpu_powers) / len(gpu_powers) if gpu_powers else None,
                    power_limit_w=samples[0].gpus[gpu_id].power_limit_w,
                ))
        
        return SystemStats(
            timestamp=samples[-1].timestamp,
            gpus=avg_gpus,
            cpu_percent=avg_cpu,
            cpu_count=samples[0].cpu_count,
            ram_used_gb=avg_ram_used,
            ram_total_gb=samples[0].ram_total_gb,
            ram_percent=avg_ram_percent,
        )
    
    def save_system_info(self, output_path: Path):
        """
        Save system hardware info to JSON.
        
        Parameters
        ----------
        output_path : Path
            Path to save system.json
        """
        info = {
            "timestamp": time.time(),
            "cpu": {
                "count": psutil.cpu_count() if PSUTIL_AVAILABLE else 0,
                "physical_cores": psutil.cpu_count(logical=False) if PSUTIL_AVAILABLE else 0,
            },
            "ram": {
                "total_gb": psutil.virtual_memory().total / (1024**3) if PSUTIL_AVAILABLE else 0,
            },
            "gpus": [],
        }
        
        if self._nvml_initialized:
            for i in range(self.gpu_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    info["gpus"].append({
                        "id": i,
                        "name": name,
                        "memory_gb": mem_info.total / (1024**3),
                    })
                except Exception as e:
                    warnings.warn(f"Error getting GPU {i} info: {e}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(info, f, indent=2)
    
    def __del__(self):
        """Cleanup."""
        self.stop()
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


if __name__ == "__main__":
    """Test hardware monitoring."""
    print("Testing hardware monitor...\n")
    
    monitor = HardwareMonitor(poll_interval=1.0)
    
    # Get current stats
    stats = monitor.get_current_stats()
    print(f"CPU: {stats.cpu_percent:.1f}%")
    print(f"RAM: {stats.ram_percent:.1f}% ({stats.ram_used_gb:.1f}/{stats.ram_total_gb:.1f} GB)")
    
    for gpu in stats.gpus:
        print(f"\nGPU {gpu.device_id}: {gpu.name}")
        print(f"  Utilization: {gpu.utilization_percent:.1f}%")
        print(f"  Memory: {gpu.memory_percent:.1f}% ({gpu.memory_used_gb:.1f}/{gpu.memory_total_gb:.1f} GB)")
        if gpu.temperature_c:
            print(f"  Temperature: {gpu.temperature_c:.1f}°C")
        if gpu.power_draw_w:
            print(f"  Power: {gpu.power_draw_w:.1f}/{gpu.power_limit_w:.1f} W")
    
    # Test background monitoring
    print("\n\nStarting 5-second background monitoring...")
    monitor.start()
    time.sleep(5)
    monitor.stop()
    
    avg_stats = monitor.get_average_stats()
    if avg_stats:
        print(f"\nAverage over 5 seconds:")
        print(f"CPU: {avg_stats.cpu_percent:.1f}%")
        print(f"RAM: {avg_stats.ram_percent:.1f}%")
        for gpu in avg_stats.gpus:
            print(f"GPU {gpu.device_id}: {gpu.utilization_percent:.1f}% util")