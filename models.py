
import torch
from torch import nn


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, num_filter_outer: int, num_filter_inner: int, num_input_channels: int, submodule: 'UnetSkipConnectionBlock', outermost: bool = False, innermost: bool = False, norm_layer: nn.Module = nn.BatchNorm2d, use_droput: bool = False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            num_filter_outer (int)          -- the number of filters in the outer conv layer
            num_filter_inner (int)          -- the number of filters in the inner conv layer
            num_input_channels (int)        -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)                -- if this module is the outermost module
            innermost (bool)                -- if this module is the innermost module
            norm_layer                      -- normalization layer
            use_dropout (bool)              -- if use dropout layers.
        """
        super().__init__()
        self.outermost = outermost

        if num_input_channels is None:
            num_input_channels = num_filter_outer

        kernel_size = 4

        downconv = nn.Conv2d(num_input_channels, num_filter_inner, kernel_size=kernel_size, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(num_filter_inner)

        uprelu = nn.ReLU(True)
        upnorm = norm_layer(num_filter_outer)

        if outermost:
            upconv = nn.ConvTranspose2d(num_filter_inner * 2, num_filter_outer, kernel_size=kernel_size, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(num_filter_inner, num_filter_outer, kernel_size=kernel_size, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(num_filter_inner * 2, num_filter_outer, kernel_size=kernel_size, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_droput:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_channels: int, output_channels: int, num_downs: int, num_filter_outer: int = 64, norm_layer: nn.Module = nn.BatchNorm2d, use_dropout: bool = False):
        """Construct a Unet generator

        Parameters:
            input_channels (int)    -- the number of channels in input images
            output_channels (int)   -- the number of channels in output images
            num_downs (int)         -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                        image of size 128x128 will become of size 1x1 # at the bottleneck
            num_filter_outer (int)  -- the number of filters in the outer conv layer
            norm_layer              -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """

        super().__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(num_filter_outer * 8, num_filter_outer * 8, None, None, innermost=True, norm_layer=norm_layer)  # add the innermost layer
        for _ in range(num_downs - 5):  # add intermediate layers with num_filter_outer * 8 filters
            unet_block = UnetSkipConnectionBlock(num_filter_outer * 8, num_filter_outer * 8, None, unet_block, use_droput=use_dropout, norm_layer=norm_layer)
        # gradually reduce the number of filters from num_filter_outer * 8 to num_filter_outer
        unet_block = UnetSkipConnectionBlock(num_filter_outer * 4, num_filter_outer * 8, None, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(num_filter_outer * 2, num_filter_outer * 4, None, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(num_filter_outer, num_filter_outer * 2, None, unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_channels, num_filter_outer, input_channels, unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_channels: int, num_filter_last_layer: int, num_layers: int):
        """Construct a PatchGAN discriminator

        Parameters:
            input_channels (int)            -- the number of channels in input images
            num_filter_last_layer (int)     -- the number of filters in the last conv layer
            num_layers (int)                -- the number of conv layers in the discriminator
        """
        super().__init__()

        kernel_size = 4
        padding = 1

        model = [
            nn.Conv2d(input_channels, num_filter_last_layer, kernel_size=kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.2, True)
        ]

        filter_scale = 1
        filter_scale_prev = 1
        for n in range(1, num_layers):
            filter_scale_prev = filter_scale
            filter_scale = min(2 ** n, 8)
            model += [
                nn.Conv2d(num_filter_last_layer * filter_scale_prev, num_filter_last_layer * filter_scale, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
                nn.BatchNorm2d(num_filter_last_layer * filter_scale),
                nn.LeakyReLU(0.2, True)
            ]

        filter_scale_prev = filter_scale
        filter_scale = min(2 ** num_layers, 8)
        model += [
            nn.Conv2d(num_filter_last_layer * filter_scale_prev, num_filter_last_layer * filter_scale, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(num_filter_last_layer * filter_scale),
            nn.LeakyReLU(0.2, True)
        ]

        model += [
            nn.Conv2d(num_filter_last_layer * filter_scale, 1, kernel_size=kernel_size, stride=1, padding=padding)
        ]  # output 1 channel prediction map

        self.model = nn.Sequential(*model)

        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode: str = 'lsgan'):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str) -- the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
        """
        super().__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))

        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError(f'gan mode {gan_mode} not implemented')

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (torch.Tensor) -- typically the prediction from a discriminator
            target_is_real (bool)     -- whether the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction: torch.Tensor, target_is_real: bool):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (torch.Tensor) -- typically the prediction output from a discriminator
            target_is_real (bool)     -- whether the ground truth label is for real images or fake images

        Returns:
            the calculated loss
        """

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
