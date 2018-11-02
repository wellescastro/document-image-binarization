import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, nb_layers):
        super(AutoEncoder, self).__init__()

        self.kernel_size = 5
        self.stride = 2
        self.nb_filters = 64

        # encoder
        self.conv_block1 = self.conv_block(in_f=1, out_f=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=2)
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=self.nb_filters, out_channels=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=2),
            nn.BatchNorm2d(self.nb_filters),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=self.nb_filters, out_channels=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=2),
            nn.BatchNorm2d(self.nb_filters),
            nn.ReLU()
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=self.nb_filters, out_channels=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=2),
            nn.BatchNorm2d(self.nb_filters),
            nn.ReLU()
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=self.nb_filters, out_channels=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=2),
            nn.BatchNorm2d(self.nb_filters),
            nn.ReLU()
        )

        #decoder
        self.deconv_block1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.nb_filters, out_channels=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=2, output_padding=1),
            nn.BatchNorm2d(self.nb_filters),
            nn.ReLU()
        )

        self.deconv_block2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.nb_filters, out_channels=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=2, output_padding=1),
            nn.BatchNorm2d(self.nb_filters),
            nn.ReLU()
        )

        self.deconv_block3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.nb_filters, out_channels=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=2, output_padding=1),
            nn.BatchNorm2d(self.nb_filters),
            nn.ReLU()
        )

        self.deconv_block4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.nb_filters, out_channels=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=2, output_padding=1),
            nn.BatchNorm2d(self.nb_filters),
            nn.ReLU()
        )

        self.deconv_block5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.nb_filters, out_channels=self.nb_filters, kernel_size=self.kernel_size, stride=self.stride, padding=2, output_padding=1),
            nn.BatchNorm2d(self.nb_filters),
            nn.ReLU()
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(self.nb_filters, 1, kernel_size=self.kernel_size, stride=1, padding=2),
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

    def forward(self, x):
        # encoding part
        # print("input", x.shape)
        x = self.conv_block1(x)
        residual_1 = x

        # print("conv_block1", x.shape)
        x = self.conv_block2(x)
        residual_2 = x

        # print("conv_block2", x.shape)
        x = self.conv_block3(x)
        residual_3 = x

        # print("conv_block3", x.shape)
        x = self.conv_block4(x)
        residual_4 = x

        # print("conv_block4", x.shape)
        x = self.conv_block5(x)
        # residual_5 = layer_5

        # print("conv_block5", x.shape)

        # print("---\n")
        # decoding part
        # x += residual_5
        x = self.deconv_block1(x)
        # print("deconv_block1", x.shape)
        x += residual_4
        x = self.deconv_block2(x)
        # print("deconv_block2", x.shape)
        x += residual_3
        x = self.deconv_block3(x)
        # print("deconv_block3", x.shape)
        x += residual_2
        x = self.deconv_block4(x)
        # print("deconv_block4", x.shape)
        x += residual_1
        x = self.deconv_block5(x)
        # print("deconv_block5", x.shape)

        x = self.output_layer(x)
        # print("output_layer", x.shape)

        return x
