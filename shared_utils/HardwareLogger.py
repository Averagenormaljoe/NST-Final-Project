import time
import psutil
import tensorflow as tf
from shared_utils.device import get_gpu_usage
class HardwareLogger:
    def __init__(self,verbose : int = 0):
        self.verbose = verbose
        self.total_wall_time = time.time() 
        self.total_time_cpu = time.process_time()
        self.start_time_cpu = time.process_time()
        self.start_time_wall = time.time()
        self.log_data = {}
    def append(self,key,item):
        self.log_data.setdefault(key, []).append(item)
    
    def log_hardware(self):
        gpu = get_gpu_usage()
        if gpu is not None:
            self.append("gpu",gpu)
        ram = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/').percent
            
        # append hardware usage statistics
        self.append("ram",ram)
        self.append("disk",disk)
        self.append("cpu",cpu)
        
        if self.verbose >= 1:
            print(f"CPU: {cpu}%, GPU: {gpu}%, RAM Usage: {ram}%")
    def log_loss(self,loss,i):
        self.append("loss",loss)
        self.append("iterations",i)
        if self.verbose > 0:
            print(f"Iteration {i}: loss={loss:.2f}")

    def record_time_diff(self):
        end_time_cpu = time.process_time()  
        end_time_wall = time.time()  
        cpu_time = end_time_cpu - self.start_time_cpu  
        wall_time = end_time_wall - self.start_time_wall
        return cpu_time, wall_time
    
    def log_process_time(self,cpu_time,wall_time):
        self.append("cpu time",cpu_time)
        self.append("wall time",wall_time)
        if self.verbose > 0:
            print(f"CPU times in seconds: {cpu_time:.2f}")
            print(f"Wall time in seconds: {wall_time:.2f}")
    
    def log_end_check(self):
        cpu_time, wall_time = self.record_time_diff()
        self.log_process_time(cpu_time,wall_time)
        self.reset_start_time()
        
    def reset_start_time(self):
        self.start_time_cpu = time.process_time()
        self.start_time_wall = time.time()
    def on_training_end(self):
        self.ending_log()
       
    def ending_log(self):
        end_time_wall = time.time()
        end_time_cpu = time.process_time()
        end_total_wall_time = end_time_wall - self.total_wall_time
        end_total_time_cpu = end_time_cpu - self.total_time_cpu
        if self.verbose > 0:
            print(f"Total wall time: {end_total_wall_time} seconds")
            print(f"Total CPU time: {end_total_time_cpu} seconds")
        
            
    def get_log(self):
        return self.log_data
    def get_name_log(self,key):
        return self.log_data.get(key, [])
        
class TFHardwareLogger(HardwareLogger):
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.start_time = None
        self.end_time = None
        self.total_duration = 0
 
    def train_end(self):
        self.end_time = time.time()
        if self.start_time is not None and self.end_time is not None:
            train_duration = self.end_time - self.start_time
            self.append("train duration",train_duration)
            if self.verbose > 0:
                print(f"Training duration: {train_duration:.2f} seconds")
            self.total_duration += train_duration
    def train_start(self):
        self.reset_tf()
        self.log_hardware()
        
    def reset_tf(self):
        self.start_time = time.time()
        self.end_time = None