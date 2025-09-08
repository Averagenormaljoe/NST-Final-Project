import subprocess
def get_gpu_usage():
    try:
        command = ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']
        return process_command(command)
    except:
        pass
   
def process_command(command):
    output = subprocess.check_output(command)
    return int(output.decode('utf-8').strip())
