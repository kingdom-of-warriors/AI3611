import torch
from torch import nn
import torchvision

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        # (batch_size, 2048, image_size/32, image_size/32)
        out = self.resnet(images)
        # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = self.adaptive_pool(out)
        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=False):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


# --- 使用示例 (可选) ---
if __name__ == '__main__':
    # 假设 encoded_image_size=14, 对应 vit_b_16 的 224x224 输入
    encoder = Encoder()
    encoder.eval() # 设置为评估模式

    # 创建一个符合 vit_b_16 期望输入的示例批次 (batch_size=4, channels=3, height=224, width=224)
    # 注意：实际使用时需要进行适当的图像预处理
    dummy_images = torch.randn(4, 3, 224, 224)

    # 前向传播
    encoded_output = encoder(dummy_images)

    # 打印输出形状 (应为 [4, 14, 14, 768])
    print("Input shape:", dummy_images.shape)
    print("Output shape:", encoded_output.shape)
