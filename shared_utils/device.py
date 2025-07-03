import subprocess
def get_gpu_usage():
    try:
        command = ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']
        return process_command(command)
    except:
        pass
   

def get_memory_usage():
    try:
        command = ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']
        return process_command(command)
    except:
        pass
    
def get_gpu_temperature():
    try:
        command = ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits']
        return process_command(command)
    except:
        pass
def get_gpu_power_usage():
    try:
        command = ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits']
        return process_command(command)
    except:
        pass

def process_command(command):
    output = subprocess.check_output(command)
    return int(output.decode('utf-8').strip())
