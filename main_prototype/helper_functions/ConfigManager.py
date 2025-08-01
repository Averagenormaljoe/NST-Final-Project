from shared_utils.HardwareLogger import HardwareLogger
class ConfigManager:
    def __init__(self, config: dict):
        self.unpack_config(config)
        self.update_settings(config)
    def unpack_config(self, config: dict):
        self.start_step = config.get("start_step", 1)
        self.iterations = config.get("iterations", 500)
        self.save_step = config.get("save_step", 100)
        self.preserve_color = config.get("preserve_color", False)
        self.noise = config.get("noise", False)
        self.clip = config.get("clip", False)
        self.chosen_optimizer = config.get("optimizer", "adam")
        self.lr = config.get("lr", 1.0)
        self.w, self.h = config.get("size", (400, 400))
        self.output_path = config.get("output_path", None)
        self.hardware_logger = HardwareLogger()
    def update_settings(self, config: dict):
        self.verbose = config.get("verbose", 0)
        self.include_checkpoints = config.get("include_checkpoints", False)