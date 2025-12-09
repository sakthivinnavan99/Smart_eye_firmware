import os
import platform
import subprocess
import psutil

def run_cmd(cmd):
    try:
        output = subprocess.check_output(cmd, shell=True, universal_newlines=True)
        return output.strip()
    except Exception as e:
        return f"Error: {e}"

def get_cpu_info():
    info = {}
    cpuinfo = run_cmd("cat /proc/cpuinfo")
    for line in cpuinfo.split('\n'):
        if "model name" in line or "Hardware" in line or "Revision" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    info["Cores"] = psutil.cpu_count(logical=False)
    info["Threads"] = psutil.cpu_count(logical=True)
    info["CPU Frequency (MHz)"] = psutil.cpu_freq().current if psutil.cpu_freq() else "Unavailable"
    return info

def get_gpu_info():
    info = {}
    info["GPU Frequency (Hz)"] = run_cmd("vcgencmd measure_clock core")
    info["GPU Temp (°C)"] = run_cmd("vcgencmd measure_temp")
    info["GPU Memory"] = run_cmd("vcgencmd get_mem gpu")
    return info

def get_ram_info():
    mem = psutil.virtual_memory()
    return {
        "Total RAM (MB)": mem.total // (1024 * 1024),
        "Available RAM (MB)": mem.available // (1024 * 1024),
        "Used RAM (MB)": mem.used // (1024 * 1024),
        "RAM Usage (%)": mem.percent
    }

def get_os_info():
    return {
        "System": platform.system(),
        "Node Name": platform.node(),
        "Release": platform.release(),
        "Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Architecture": platform.architecture()[0],
        "Distribution": run_cmd("cat /etc/os-release | grep PRETTY_NAME").split("=")[-1].strip('"')
    }

def main():
    print("=== Raspberry Pi System Info ===\n")

    print("\n--- CPU Info ---")
    for k, v in get_cpu_info().items():
        print(f"{k}: {v}")

    print("\n--- GPU Info ---")
    for k, v in get_gpu_info().items():
        print(f"{k}: {v}")

    print("\n--- RAM Info ---")
    for k, v in get_ram_info().items():
        print(f"{k}: {v}")

    print("\n--- OS Info ---")
    for k, v in get_os_info().items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
