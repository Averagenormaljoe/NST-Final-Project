from itertools import product
list_of_optimizers = ["adam", "sgd", "rmsprop", "adagrad", "adamax"]
list_of_loss_networks = ["vgg19", "vgg16", "mobilenet", "resnet50", "inception_v3"]
enable_clip = [True, False]
use_l2 = [True, False]
list_of_image_sizes = [(32,32),(64,64),(128,128),(256, 400), (512, 600), (1024, 800), (2048, 1200)]
list_of_content_weights = [1e-6, 2.5e-8, 1e-4,]
list_of_style_weights = [1e-6, 0.8e-6, 0.5e-6, 0.3e-6, 0.1e-6, 0.1e-7]
list_of_total_variation_weights = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, ]
list_of_ssim_weights = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.1]
list_of_psnr_weights = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.1]
list_of_lpips_weights = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.1]
list_of_learning_rates = [0.1,0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
list_of_iterations = [100, 200, 300, 400, 500]
list_of_check_steps = [10, 20, 50, 100]
list_of_noise_strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
list_of_improvement_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
list_of_beta_1s = [0.9, 0.99, 0.999]
list_of_epsilons = [1e-1, 1e-2, 1e-3]
list_of_decay_steps = [100, 200, 300]
list_of_decay_rates = [0.96, 0.98, 0.99]
list_of_weight_decays = [1e-4, 1e-5, 1e-6]
lpips_loss_nets = ["alex", "vgg", "squeeze"]

def get_hyperparameters():
    return {
        "optimizers": list_of_optimizers,
        "loss_networks": list_of_loss_networks,
        "enable_clip": enable_clip,
        "use_l2": use_l2,
        "image_sizes": list_of_image_sizes,
        "content_weights": list_of_content_weights,
        "style_weights": list_of_style_weights,
        "total_variation_weights": list_of_total_variation_weights,
        "ssim_weights": list_of_ssim_weights,
        "psnr_weights": list_of_psnr_weights,
        "lpips_weights": list_of_lpips_weights,
        "learning_rates": list_of_learning_rates,
        "iterations": list_of_iterations,
        "check_steps": list_of_check_steps,
        "noise_strengths": list_of_noise_strengths,
        "improvement_thresholds": list_of_improvement_thresholds,
        "beta_1s": list_of_beta_1s,
        "epsilons": list_of_epsilons,
        "decay_steps": list_of_decay_steps,
        "decay_rates": list_of_decay_rates,
        "weight_decays": list_of_weight_decays,
        "lpips_loss_nets": lpips_loss_nets
    }
def grid_search_hyperparameters():
    hyperparameters = get_hyperparameters()
