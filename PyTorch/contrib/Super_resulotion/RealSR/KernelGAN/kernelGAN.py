import torch
import loss
import networks
import torch.nn.functional as F
from util import save_final_kernel, run_zssr, post_process_k


class KernelGAN:
    # Constraint co-efficients
    lambda_sum2one = 0.5
    lambda_bicubic = 5
    lambda_boundaries = 0.5
    lambda_centralized = 0
    lambda_sparse = 0

    def __init__(self, conf):
        # Acquire configuration
        self.conf = conf

        # Define the GAN
        self.G = networks.Generator(conf).sdaa()
        self.D = networks.Discriminator(conf).sdaa()

        # Calculate D's input & output shape according to the shaving done by the networks
        self.d_input_shape = self.G.output_size
        self.d_output_shape = self.d_input_shape - self.D.forward_shave

        # Input tensors
        self.g_input = torch.FloatTensor(1, 3, conf.input_crop_size, conf.input_crop_size).sdaa()
        self.d_input = torch.FloatTensor(1, 3, self.d_input_shape, self.d_input_shape).sdaa()

        # The kernel G is imitating
        self.curr_k = torch.FloatTensor(conf.G_kernel_size, conf.G_kernel_size).sdaa()

        # Losses
        self.GAN_loss_layer = loss.GANLoss(d_last_layer_size=self.d_output_shape).sdaa()
        self.bicubic_loss = loss.DownScaleLoss(scale_factor=conf.scale_factor).sdaa()
        self.sum2one_loss = loss.SumOfWeightsLoss().sdaa()
        self.boundaries_loss = loss.BoundariesLoss(k_size=conf.G_kernel_size).sdaa()
        self.centralized_loss = loss.CentralizedLoss(k_size=conf.G_kernel_size, scale_factor=conf.scale_factor).sdaa()
        self.sparse_loss = loss.SparsityLoss().sdaa()
        self.loss_bicubic = 0

        # Define loss function
        self.criterionGAN = self.GAN_loss_layer.forward

        # Initialize networks weights
        self.G.apply(networks.weights_init_G)
        self.D.apply(networks.weights_init_D)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=conf.g_lr, betas=(conf.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=conf.d_lr, betas=(conf.beta1, 0.999))

        print('*' * 60 + '\nSTARTED KernelGAN on: \"%s\"...' % conf.input_image_path)

    # noinspection PyUnboundLocalVariable
    def calc_curr_k(self):
        """given a generator network, the function calculates the kernel it is imitating"""
        delta = torch.Tensor([1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).sdaa()
        for ind, w in enumerate(self.G.parameters()):
            curr_k = F.conv2d(delta, w, padding=self.conf.G_kernel_size - 1) if ind == 0 else F.conv2d(curr_k, w)
        self.curr_k = curr_k.squeeze().flip([0, 1])

    def train(self, g_input, d_input):
        self.set_input(g_input, d_input)
        self.train_g()
        self.train_d()

    def set_input(self, g_input, d_input):
        self.g_input = g_input.contiguous()
        self.d_input = d_input.contiguous()

    def train_g(self):
        # Zeroize gradients
        self.optimizer_G.zero_grad()
        # Generator forward pass
        g_pred = self.G.forward(self.g_input)
        # Pass Generators output through Discriminator
        d_pred_fake = self.D.forward(g_pred)
        # Calculate generator loss, based on discriminator prediction on generator result
        loss_g = self.criterionGAN(d_last_layer=d_pred_fake, is_d_input_real=True)
        # Sum all losses
        total_loss_g = loss_g + self.calc_constraints(g_pred)
        # Calculate gradients
        total_loss_g.backward()
        # Update weights
        self.optimizer_G.step()

    def calc_constraints(self, g_pred):
        # Calculate K which is equivalent to G
        self.calc_curr_k()
        # Calculate constraints
        self.loss_bicubic = self.bicubic_loss.forward(g_input=self.g_input, g_output=g_pred)
        loss_boundaries = self.boundaries_loss.forward(kernel=self.curr_k)
        loss_sum2one = self.sum2one_loss.forward(kernel=self.curr_k)
        loss_centralized = self.centralized_loss.forward(kernel=self.curr_k)
        loss_sparse = self.sparse_loss.forward(kernel=self.curr_k)
        # Apply constraints co-efficients
        return self.loss_bicubic * self.lambda_bicubic + loss_sum2one * self.lambda_sum2one + \
               loss_boundaries * self.lambda_boundaries + loss_centralized * self.lambda_centralized + \
               loss_sparse * self.lambda_sparse

    def train_d(self):
        # Zeroize gradients
        self.optimizer_D.zero_grad()
        # Discriminator forward pass over real example
        d_pred_real = self.D.forward(self.d_input)
        # Discriminator forward pass over fake example (generated by generator)
        # Note that generator result is detached so that gradients are not propagating back through generator
        g_output = self.G.forward(self.g_input)
        d_pred_fake = self.D.forward((g_output + torch.randn_like(g_output) / 255.).detach())
        # Calculate discriminator loss
        loss_d_fake = self.criterionGAN(d_pred_fake, is_d_input_real=False)
        loss_d_real = self.criterionGAN(d_pred_real, is_d_input_real=True)
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        # Calculate gradients, note that gradients are not propagating back through generator
        loss_d.backward()
        # Update weights, note that only discriminator weights are updated (by definition of the D optimizer)
        self.optimizer_D.step()

    def finish(self):
        final_kernel = post_process_k(self.curr_k, n=self.conf.n_filtering)
        save_final_kernel(final_kernel, self.conf)
        print('KernelGAN estimation complete!')
        run_zssr(final_kernel, self.conf)
        print('FINISHED RUN (see --%s-- folder)\n' % self.conf.output_dir_path + '*' * 60 + '\n\n')
