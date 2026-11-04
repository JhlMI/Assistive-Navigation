"""
monitor.py — Monitor de recursos multiplataforma
PC     : pynvml (GPU discreta)
Jetson : jtop   (GPU integrada, memoria unificada)
"""
import platform
import time

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

IS_JETSON = "aarch64" in platform.machine()

if IS_JETSON:
    try:
        from jtop import jtop
        JTOP_OK = True
    except ImportError:
        JTOP_OK = False
    NVML_OK = False
else:
    JTOP_OK = False
    try:
        import pynvml
        pynvml.nvmlInit()
        NVML_OK = True
    except Exception:
        NVML_OK = False


class Monitor:
    def __init__(self):
        if PSUTIL_OK:
            psutil.cpu_percent()

        if IS_JETSON and JTOP_OK:
            self._jetson      = jtop()
            self._jetson.start()
            time.sleep(0.5)
            self.gpu_nombre   = "Jetson Orin Nano"
            self.gpu_total_mb = 8192
            self._modo        = "jetson"
        elif NVML_OK:
            self._handle      = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem               = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self.gpu_nombre   = pynvml.nvmlDeviceGetName(self._handle)
            self.gpu_total_mb = mem.total / 1e6
            self._modo        = "nvml"
        else:
            self.gpu_nombre   = "GPU (no disponible)"
            self.gpu_total_mb = 0
            self._modo        = "none"

        print(f"  Monitor: modo={self._modo}  GPU={self.gpu_nombre}")

    def leer(self) -> dict:
        cpu    = psutil.cpu_percent() if PSUTIL_OK else 0
        ram_gb = psutil.virtual_memory().used / 1e9 if PSUTIL_OK else 0

        if self._modo == "jetson" and JTOP_OK:
            stats   = self._jetson.stats
            gpu_pct = float(stats.get("GPU", 0))
            ram_tot = float(stats.get("RAM", 0))
            ram_gb  = ram_tot * self.gpu_total_mb / 1e3
            vram_mb = ram_gb * 1024
            # Power TOT viene en mW — convertir a W
            power_mw = float(stats.get("Power TOT", 0))
            power_w  = round(power_mw / 1000.0, 2)
            # Temperatura GPU
            temp_gpu = float(stats.get("Temp gpu", 0))
            return {
                "ram_gb":   round(ram_gb, 2),
                "cpu_pct":  cpu,
                "gpu_mb":   round(vram_mb, 1),
                "gpu_pct":  round(gpu_pct, 1),
                "power_w":  power_w,
                "temp_c":   temp_gpu,
            }

        elif self._modo == "nvml":
            mem  = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            return {
                "ram_gb":  round(ram_gb, 2),
                "cpu_pct": cpu,
                "gpu_mb":  round(mem.used / 1e6, 1),
                "gpu_pct": util.gpu,
                "power_w": 0.0,
                "temp_c":  0.0,
            }

        return {"ram_gb": round(ram_gb,2), "cpu_pct": cpu,
                "gpu_mb": 0, "gpu_pct": 0, "power_w": 0.0, "temp_c": 0.0}

    def info_sistema(self) -> dict:
        import torch
        vm = psutil.virtual_memory() if PSUTIL_OK else None
        return {
            "os":           platform.system()+" "+platform.release(),
            "plataforma":   "Jetson" if IS_JETSON else "PC",
            "python":       platform.python_version(),
            "cpu":          platform.processor() or platform.machine(),
            "ram_total_gb": round(vm.total/1e9, 1) if vm else 0,
            "gpu":          self.gpu_nombre,
            "gpu_vram_gb":  round(self.gpu_total_mb/1e3, 1),
            "cuda":         torch.version.cuda,
            "pytorch":      torch.__version__,
        }

    def cleanup(self):
        if self._modo == "jetson" and JTOP_OK:
            self._jetson.close()
