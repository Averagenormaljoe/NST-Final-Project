import subprocess
def get_gpu_usage():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']
        )
        return int(output.decode('utf-8').strip())
    except:
        return None
