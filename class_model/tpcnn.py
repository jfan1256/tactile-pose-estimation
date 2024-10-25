import torch
import numpy as np
import torch.nn as nn

from exec_train.utils_space import remove_small, check_link

# TPCNN
class TPCNN(nn.Module):
    # Initialize TPCNN
    def __init__(self, configs):
        super(TPCNN, self).__init__()

        # ***************************************************Encoder***************************************************
        self.conv_enc_1 = nn.Sequential(
            nn.Conv2d(2 * configs['window_size'], 32, kernel_size=(3,3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))
        self.conv_enc_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2))
        self.conv_enc_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.conv_enc_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2))
        self.conv_enc_5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512))
        self.conv_enc_6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(5,5)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024))
        self.conv_enc_7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(3,3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2))

        # ***************************************************Decoder***************************************************
        self.conv_dec_1 = nn.Sequential(
            nn.Conv3d(1025, 1025, kernel_size=(3,3,3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(1025))
        self.conv_dec_2 = nn.Sequential(
            nn.Conv3d(1025, 512, kernel_size=(3,3,3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(512))
        self.conv_dec_3 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=(2,2,2), stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm3d(256))
        self.conv_dec_4 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=(3,3,3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(128))
        self.conv_dec_5 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(3,3,3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(64))
        self.conv_dec_6 = nn.Conv3d(64, 21, kernel_size=(3,3,3), padding=1)

        # # ***************************************************TEST DATA***************************************************
        self.height, self.width, self.depth, self.channel = configs.height, configs.width, configs.depth, configs.channel
        pos_y, pos_x, pos_z = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height), np.linspace(0, 1, self.depth))
        self.pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width * self.depth)).float()
        self.pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width * self.depth)).float()
        self.pos_z = torch.from_numpy(pos_z.reshape(self.height * self.width * self.depth)).float()

    # Encoder
    def encoder(self, tactile):
        # tactile.shape = (batch_size, window_size, width=96, height=96)
        output = self.conv_enc_1(tactile)  # --> output.shape = (batch_size, num_kernel=32, width=96, height=96)
        output = self.conv_enc_2(output)  # --> output.shape = (batch_size, num_kernel=64, width=48, height=48)
        output = self.conv_enc_3(output)  # --> output.shape = (batch_size, num_kernel=128, width=48, height=48)
        output = self.conv_enc_4(output)  # --> output.shape = (batch_size, num_kernel=256, width=24, height=24)
        output = self.conv_enc_5(output)  # --> output.shape = (batch_size, num_kernel=512, width=24, height=24)
        output = self.conv_enc_6(output)  # --> output.shape = (batch_size, num_kernel=1024, width=20, height=20)
        output = self.conv_enc_7(output)  # --> output.shape = (batch_size, num_kernel=1024, width=10, height=10)

        # Repeat the output 9 times
        output = output.reshape(output.shape[0], output.shape[1], output.shape[2], output.shape[3], 1)
        output = output.repeat(1, 1, 1, 1, 9)  # --> output.shape = (batch_size, num_kernel=1024, width=10, height=10, depth=9)

        # Add a positional one-hot encoded layer to kernel
        pos_layer = torch.zeros(output.shape[0], 1, output.shape[2], output.shape[3], output.shape[4]).to(self.configs.device)
        for i in range(pos_layer.shape[4]):
            pos_layer[:, :, :, :, i] = i
        pos_layer = pos_layer / (pos_layer.shape[4] - 1)
        output = torch.cat((output, pos_layer), axis=1)  # --> output.shape = (batch_size, num_kernel=1025, width=10, height=10, depth=9)

        # Return
        return output

    # Decoder
    def decoder(self, output):
        # output.shape = (batch_size, num_kernel=1025, height=10, width=10, depth=9)
        output = self.conv_dec_1(output)  # --> output.shape = (batch_size, num_kernel=1025, width=10, height=10, depth=9)
        output = self.conv_dec_2(output)  # --> output.shape = (batch_size, num_kernel=512, width=10, height=10, depth=9)
        output = self.conv_dec_3(output)  # --> output.shape = (batch_size, num_kernel=256, width=20, height=20, depth=18)
        output = self.conv_dec_4(output)  # --> output.shape = (batch_size, num_kernel=128, width=20, height=20, depth=18)
        output = self.conv_dec_5(output)  # --> output.shape = (batch_size, num_kernel=64, width=20, height=20, depth=18)
        output = self.conv_dec_6(output)  # --> output.shape = (batch_size, num_kernel=21, width=20, height=20, depth=18)

        # Convert to probabilities for heatmap
        prob = torch.sigmoid(output)  # --> prob.shape = (batch_size, channel=21, width=20, height=20, depth=18)
        heatmap_out = prob.reshape(-1, 21, 20, 20, 18)  # --> heatmap_out.shape = (batch_size, channel=21, width=20, height=20, depth=18)
        heatmap_transform = remove_small(heatmap_out.transpose(2, 3), 1e-2, self.configs.device)  # --> heatmap_transform.shape = (batch_size, channel=21, height=20, width=20, depth=18)

        # Return
        return heatmap_transform

    # Keypoint
    def keypoint(self, heatmap_transform):
        # Calculate expected keypoint
        softmax_attention = heatmap_transform.reshape(-1, self.height * self.width * self.depth) # --> softmax_attention.shape = (batch_size * channel, height * width * depth)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True) / (torch.sum(softmax_attention, dim=1, keepdim=True) + 1e-6) # --> expected_x.shape = (batch_size * channel, 1)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True) / (torch.sum(softmax_attention, dim=1, keepdim=True) + 1e-6) # --> expected_y.shape = (batch_size * channel, 1)
        expected_z = torch.sum(self.pos_z * softmax_attention, dim=1, keepdim=True) / (torch.sum(softmax_attention, dim=1, keepdim=True) + 1e-6) # --> expected_z.shape = (batch_size * channel, 1)
        expected_xyz = torch.cat([expected_x, expected_y, expected_z], 1) # --> expected_xyz.shape = (batch_size * channel, 3)
        keypoint_out = expected_xyz.reshape(-1, self.channel, 3) # --> keypoint_out.shape = (batch_size, channel, 3)

        # Return
        return keypoint_out

    # Forward pass of TPCNN
    def forward(self, tactile, heatmap, keypoint):
        # ************************Encoder************************
        # tactile.shape = (batch_size, window_size, width=96, height=96)
        output = self.encoder(tactile)

        # ************************Decoder************************
        # output.shape = (batch_size, num_kernel=1025, height=10, width=10, depth=9)
        heatmap_transform = self.decoder(output)

        # ************************Keypoint************************
        # heatmap_transform.shape = (batch_size, channel=21, height=20, width=20, depth=18)
        keypoint_out = self.keypoint(heatmap_transform)

        # Calculate loss
        # keypoint.shape = (batch_size, channel, 3)
        loss_keypoint = nn.MSELoss()(keypoint_out, keypoint)
        loss_link = torch.mean(check_link(self.configs.link_min, self.configs.link_max, keypoint_out, self.configs.device)) * 10

        # Return losses
        return {
            'loss_link': loss_link,
            'loss_keypoint': loss_keypoint
        }

    # Generate
    def generate(self, tactile, heatmap, keypoint):
        # ************************Encoder************************
        # tactile.shape = (batch_size, window_size, width=96, height=96)
        output = self.encoder(tactile)

        # ************************Decoder************************
        # output.shape = (batch_size, num_kernel=1025, height=10, width=10, depth=9)
        heatmap_transform = self.decoder(output)

        # ************************Keypoint************************
        # heatmap_transform.shape = (batch_size, channel=21, height=20, width=20, depth=18)
        keypoint_out = self.keypoint(heatmap_transform)

        return heatmap_transform, keypoint_out