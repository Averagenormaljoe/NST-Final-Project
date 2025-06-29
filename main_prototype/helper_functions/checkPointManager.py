from shared_utils.helper import create_dir
import tensorflow as tf

class checkPointManager:
    def __init__(self, combination_image, optimizer : str, checkpoint_dir : str, folder_path : str):
        self.optimizer = optimizer
        self.combination_image = combination_image
        self.checkpoint_dir = checkpoint_dir
        self.folder_path =  folder_path
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, combination_image=combination_image)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=checkpoint_dir, max_to_keep=5)
    def get_manager(self):
        return self.manager.save()
    def save_checkpoint(self, step,checkpoint_prefix):
        self.checkpoint.save(file_prefix=checkpoint_prefix)
    def setup(self):
        create_dir(self.folder_path)
        checkpoint_dir = "./checkpoints"
        create_dir(checkpoint_dir)