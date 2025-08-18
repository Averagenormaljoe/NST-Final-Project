
import math
import os

import keras
import tensorflow as tf
from helper_functions.training_helper import result_save
from tqdm import trange

from gatys_functions.apply_style_transfer_step import apply_style_transfer_step
from gatys_functions.preprocess_NST_images import preprocess_NST_images
from helper_functions.ConfigManager import ConfigManager
from helper_functions.bestImage import BestImage
from helper_functions.helper import deprocess_image
from helper_functions.log_funcs import create_log_dir
from shared_utils.optimizer import get_optimizer
from video_utils.helper.process_flow_on_frames import process_flow_on_frames
class LoopManager(ConfigManager):
    def __init__(self, config: dict):
        super().__init__(config)
    def should_save(self, step: int) -> bool:
        return step % self.save_step == 0 or step == self.iterations
    def invalid_iterations(self):
        if self.start_step > self.iterations:
            print(f"Start step ({self.start_step}) is greater than the specified iterations ({self.iterations}). No training will be performed.")
            return True
    def return_error(self):
        return [], BestImage(-1,-1,-1),{}
    def get_optimizer(self,string_optimizer):
        if isinstance(string_optimizer, str):
            optimizer = get_optimizer(string_optimizer, learning_rate=self.lr)
            return optimizer
        else:
            print(f"Invalid passed in optimizer type: {type(self.string_optimizer)}. Should be a string.\n")
            return None
    def update_optimizer(self, optimizer, config):
        if optimizer is None:
            return None
        lr = config.get("learning_rate", self.lr)
        optimizer.learning_rate = lr
        return optimizer
        
    def training_loop(self,content_path, style_path,content_name : str = "",style_name: str = "",config : dict={},device_config : dict = {}):
        if not os.path.exists(content_path) and not os.path.exists(style_path):
            raise FileNotFoundError("Both of the paths for the style or content images are invalid.")
        if not os.path.exists(content_path):
            raise FileNotFoundError(f"Content image path does not exist.")
        if not os.path.exists(style_path):
            raise FileNotFoundError(f"Style image path does not exist.")
        self.unpack_config(config)
        base_image, style_image, combination_image = preprocess_NST_images(
                    content_path, style_path,config,device_config)
        generated_images = []
        optimizer = self.get_optimizer(self.string_optimizer)
        optimizer = self.update_optimizer(optimizer, config)
        if optimizer is None or self.invalid_iterations():
            return self.return_error()
        best_cost = math.inf
        best_image = None
        log_dir = create_log_dir(content_name, style_name)
        file_writer = tf.summary.create_file_writer(log_dir)
        name : str = f"({content_name}) + ({style_name})"
        save_image = config.get("save_image", True)
        output_path = config.get("output_path", None)
        is_checkpoint = config.get("is_checkpoint", False)
        
        if is_checkpoint:
            checkpoint_dir = os.path.join(log_dir, "tf_checkpoints")
            ckpt = tf.train.Checkpoint(optimizer=optimizer, combination_image=combination_image)
            manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
            latest_checkpoint = manager.latest_checkpoint
            if latest_checkpoint:
                ckpt.restore(latest_checkpoint)
                print(f"Restored from {latest_checkpoint}")
            else:
                print("Initializing from scratch.")
        
        config = process_flow_on_frames(config, combination_image=combination_image)
        for i in trange(self.start_step, self.iterations + 1, desc=f"{name} NST Optimization Loop Progress", disable=not self.verbose):
            loss, grads,optimizer, all_metrics,metrics_dict = apply_style_transfer_step(combination_image, base_image, style_image, optimizer,config,device_config)
            if self.should_save(i):
                # hardware usage
                float_loss = float(loss)
                img = deprocess_image(combination_image.numpy(), self.w, self.h)
                self.log_metrics(all_metrics, base_image, combination_image,metrics_dict=metrics_dict)
                self.log_save(float_loss, i)
                
                if loss < best_cost:
                    best_cost = loss
                    best_image = BestImage(img, best_cost, i)
                generated_images.append(img)
                if save_image:
                    result_save(content_name, style_name, i, img)
                with file_writer.as_default():
                    tf.summary.scalar("loss", float_loss, step=i)
                    tf.summary.image("generated_image", combination_image, step=i)
                if is_checkpoint:
                    manager.save(checkpoint_number=i)
        if output_path is not None and best_image is not None:
            keras.utils.save_img(output_path, best_image.get_image()) 
            
        file_writer.close()        
        log_data = self.end_training()
        return generated_images, best_image,log_data