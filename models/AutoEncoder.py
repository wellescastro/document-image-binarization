import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, nb_layers):
        super(AutoEncoder, self).__init__()

        self.kernel_size = 5
        self.stride = 2
        self.nb_filters = 64
        self.padding = 2

        # encoder
        self.conv_block1 = self.conv_block(in_f=1, out_f=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=2)
        self.conv_block2 = self.conv_block(in_f=self.nb_filters, out_f=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.conv_block3 = self.conv_block(in_f=self.nb_filters, out_f=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.conv_block4 = self.conv_block(in_f=self.nb_filters, out_f=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.conv_block5 = self.conv_block(in_f=self.nb_filters, out_f=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        #decoder
        self.deconv_block1 = self.deconv_block(in_f=self.nb_filters, out_f=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=1)
        self.deconv_block2 = self.deconv_block(in_f=self.nb_filters, out_f=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=1)
        self.deconv_block3 = self.deconv_block(in_f=self.nb_filters, out_f=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=1)
        self.deconv_block4 = self.deconv_block(in_f=self.nb_filters, out_f=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=1)
        self.deconv_block5 = self.deconv_block(in_f=self.nb_filters, out_f=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(self.nb_filters, 1, kernel_size=self.kernel_size, stride=1, padding=self.padding=),
            nn.Sigmoid()
        )

    def _get_padding(self, size, kernel_size=3, stride=1, dilation=1):
        padding = ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) // 2
        return padding

    def conv_block(self, in_f, out_f, *args, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, *args, **kwargs),
            nn.BatchNorm2d(out_f),
            nn.ReLU()
        )

    def deconv_block(self, in_f, out_f, *args, **kwargs):
        return nn.Sequential(
            nn.ConvTranspose2d(in_f, out_f, *args, **kwargs),
            nn.BatchNorm2d(out_f),
            nn.ReLU()
        )

    def forward(self, x):
        # encoding part
        x = self.conv_block1(x)
        residual_1 = x

        x = self.conv_block2(x)
        residual_2 = x

        x = self.conv_block3(x)
        residual_3 = x

        x = self.conv_block4(x)
        residual_4 = x

        x = self.conv_block5(x)
        residual_5 = x

        # decoding part
        x += residual_5
        x = self.deconv_block1(x)

        x += residual_4
        x = self.deconv_block2(x)

        x += residual_3
        x = self.deconv_block3(x)

        x += residual_2
        x = self.deconv_block4(x)

        x += residual_1
        x = self.deconv_block5(x)

        x = self.output_layer(x)

        return x
