from keras.saving import register_keras_serializable
from shared_utils.HardwareLogger import TFHardwareLogger
import keras
import tensorflow as tf
from video_utils.mask import temporal_warping_error
from AdaIN_functions.ada_in import ada_in, get_mean_std
@register_keras_serializable()
class NeuralStyleTransfer(tf.keras.Model):
    def __init__(self, encoder, decoder, loss_net, style_weight,channels, att = True, **kwargs):
        super(NeuralStyleTransfer, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss_net = loss_net
        self.style_weight = style_weight
        self.is_log = False
        self.include_custom_metrics = True
        self.hardwareLogger = TFHardwareLogger()
        self.channels = channels
        self.att = att
        floor_C = channels // 2
        self.f_layer = keras.layers.Conv2D(floor_C, kernel_size=1, padding='same',name="f_conv")
        self.g_layer = keras.layers.Conv2D(floor_C, kernel_size=1, padding='same', name="g_conv")
        self.h_layer = keras.layers.Conv2D(channels, kernel_size=1, padding='same', name="h_conv")
    def train_start(self):
        if self.is_log:
            self.hardwareLogger.train_start()
    def train_end(self):
        if self.is_log:
            self.hardwareLogger.train_end()
        
    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.style_loss_tracker = keras.metrics.Mean(name="style_loss")
        self.content_loss_tracker = keras.metrics.Mean(name="content_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.tv_loss_tracker = keras.metrics.Mean(name="tv_loss")
    def reconstruct_image(self, style, content):
        # Encode the style and content image.
        style_encoded = self.encoder(style)
        content_encoded = self.encoder(content)
        # Compute the AdaIN target feature maps.
        t = self.layers_ada_in(style=style_encoded, content=content_encoded)
        # Generate the neural style transferred image.
        reconstructed_image = self.decoder(t)
        return reconstructed_image,t



    def get_results(self):
        return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
            "tv_loss": self.tv_loss_tracker.result(),
    

        }


    def all_update_state(self, style_loss : float, content_loss : float, total_loss : float, tv_loss : float):
        # Update the trackers.
        self.style_loss_tracker.update_state(style_loss)
        self.content_loss_tracker.update_state(content_loss)
        self.total_loss_tracker.update_state(total_loss)
        # new metrics
        self.tv_loss_tracker.update_state(tv_loss)
    def apply_optimizer(self, gradients,trainable_vars):
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    def compute_gradients(self,total_loss, tape):
        trainable_vars = self.decoder.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.apply_optimizer(gradients, trainable_vars)
        return gradients
    def calculate_temporal_loss(self, prev_stylized_frame, curr_stylized_frame,flow, mask=None):
        temporal_loss = temporal_warping_error(prev_stylized_frame, curr_stylized_frame, flow,mask)
        return temporal_loss
    def compute_loop_loss_style(self, mean_inp, mean_out, std_inp, std_out):
        loss_style = self.loss_fn(mean_inp, mean_out) + self.loss_fn(std_inp, std_out)
        return loss_style
    def compute_total_variation(self,reconstructed_image):
        total_variation_loss = tf.reduce_mean(tf.image.total_variation(reconstructed_image))
        return total_variation_loss
    def compute_total_loss(self, loss_content : float,loss_style : float, tv_loss : float) -> float:
        total_loss = loss_content + loss_style + 1e-6 * tv_loss
        return total_loss
    def compute_loss_style(self, style_vgg_features, recons_vgg_features):
        # Initialize the style loss.
        loss_style = 0.0
        for inp, out in zip(style_vgg_features, recons_vgg_features):
            mean_inp, std_inp = get_mean_std(inp)
            mean_out, std_out = get_mean_std(out)
            loss_style += self.compute_loop_loss_style(mean_inp,mean_out,std_inp,std_out)
        return loss_style
    def compute_losses(self, style, t, reconstructed_image):
        # Initialize the content loss.
        loss_content = 0.0
        # Compute the losses.
        recons_vgg_features = self.loss_net(reconstructed_image)
        style_vgg_features = self.loss_net(style)
        loss_content = self.loss_fn(t, recons_vgg_features[-1])
        loss_style = self.compute_loss_style(style_vgg_features, recons_vgg_features)
        loss_style = self.style_weight * loss_style
        return loss_content,loss_style

    def generate_image_and_losses(self,style,content):
        # Reconstruct the image
        reconstructed_image,t = self.reconstruct_image(style, content)
        # Compute the losses.
        loss_content, loss_style = self.compute_losses(style, t, reconstructed_image)
        return reconstructed_image, loss_content, loss_style,t

    def process_inputs(self,inputs):
        # Unpack the inputs.
        style, content = inputs
        return style,content

    def log_hardware(self):
        if self.is_log:
            self.hardwareLogger.log_hardware()

    def train_step(self, inputs):

        self.train_start()
        style, content = inputs
        with tf.GradientTape() as tape:
            # Reconstruct the image.
            reconstructed_image,loss_content, loss_style,t = self.generate_image_and_losses(style,content)
            tv_loss = self.compute_total_variation(reconstructed_image)
            # Compute the total variation loss.
            total_loss = self.compute_total_loss(loss_content, loss_style, tv_loss)

        # Compute gradients and optimize the decoder.
        gradients = self.compute_gradients(total_loss, tape)
        # Update the trackers.
        self.all_update_state(loss_style, loss_content, total_loss, tv_loss)

        # metrics

        self.log_hardware()
        self.train_end()

        return self.get_results()
    def test_step(self, inputs):
        style, content = inputs
        # Reconstruct the image.
        reconstructed_image,loss_content, loss_style,t = self.generate_image_and_losses(style,content)
        # Compute the losses.
        loss_content, loss_style = self.compute_losses(style, t, reconstructed_image)
        # Compute the total variation loss and other metrics
        tv_loss = self.compute_total_variation(reconstructed_image)
        # Compute the total loss.
        total_loss = self.compute_total_loss(loss_content, loss_style, tv_loss)
        # Update the trackers.
        self.all_update_state(loss_style, loss_content, total_loss, tv_loss,)
        return self.get_results()
    def get_resource_stats(self):
        return {
            "cpu": self.hardwareLogger.get_name_log("cpu"),
            "gpu": self.hardwareLogger.get_name_log("gpu"),
            "ram" : self.hardwareLogger.get_name_log("ram"),
            "disk": self.hardwareLogger.get_name_log("disk")
        }
    def execute_stylization(self,inputs):
        style = inputs[0]
        content = inputs[1]
        style_encoded = self.encoder(style)
        content_encoded = self.encoder(content)
        t = self.layers_ada_in(style=style_encoded, content=content_encoded)
        reconstructed_image = self.decoder(t)
        return reconstructed_image
    
    def layers_ada_in(self, style, content):
        layers = (self.f_layer, self.g_layer, self.h_layer)
        t = ada_in(style, content,layers, self.att)
        return t
 
    def avg_multi_NST(self, style_images, content_encoded):
        stylized_images = []
        for s in style_images:
            style_encoded = self.encoder(s)
            res = self.layers_ada_in(style=style_encoded, content=content_encoded)
            stylized_images.append(res)
        
        blend = tf.reduce_mean(stylized_images)
        reconstructed_image = self.decoder(blend)
        return reconstructed_image
    def call(self, inputs):
        style_image = inputs[0]
        content_image = inputs[1]
        reconstructed_image = self.execute_stylization(inputs=(style_image, content_image))
        return reconstructed_image
    def build(self, input_shape):
        pass
    
    
    @property
    def metrics(self):
        return [
            self.style_loss_tracker,
            self.content_loss_tracker,
            self.total_loss_tracker,
            self.tv_loss_tracker,

        ]
