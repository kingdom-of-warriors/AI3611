import torch
from torch import nn
import torchvision

class ViTEncoder(nn.Module):
    """
    基于 Vision Transformer (ViT) 的图像编码器。
    """
    def __init__(self, encoded_image_size=14, model_name='vit_b_16', pretrained=True):
        """
        初始化 ViT 编码器。

        :param encoded_image_size: 编码后图像网格的大小 (应等于 sqrt(num_patches))。
                                   对于 vit_b_16 和 224x224 输入，此值应为 14。
        :param model_name: 要使用的 torchvision ViT 模型名称 (例如 'vit_b_16')。
        :param pretrained: 是否加载预训练权重。
        """
        super(ViTEncoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # 加载预训练的 ViT 模型
        # 使用 get_model 以便将来更容易切换模型
        self.vit = torchvision.models.get_model(model_name, weights='DEFAULT' if pretrained else None)

        # 获取 ViT 的特征维度 (embedding dimension)
        self.feature_dim = self.vit.hidden_dim

        # 移除原始的分类头 (替换为一个恒等映射)
        self.vit.heads = nn.Identity()


        # 初始化微调设置 (默认冻结所有参数)
        self.fine_tune(fine_tune=False)

    def forward(self, images):
        """
        前向传播。

        :param images: 输入图像张量, 形状为 (batch_size, 3, image_size, image_size)。
                       image_size 需要与 ViT 模型和 encoded_image_size 兼容。
        :return: 编码后的图像特征, 形状为 (batch_size, encoded_image_size, encoded_image_size, feature_dim)。
        """
        batch_size = images.shape[0]

        # 输入: (batch_size, 3, H, W)
        # 输出: (batch_size, num_patches, feature_dim)
        x = self.vit.conv_proj(images)
        x = x.flatten(2).transpose(1, 2)

        # 获取 class token (形状: [1, 1, feature_dim]) 并扩展以匹配 batch size
        class_token = self.vit.class_token.expand(batch_size, -1, -1)
        # 输出形状: (batch_size, num_patches + 1, feature_dim)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.vit.encoder.pos_embedding
        encoder_output = self.vit.encoder.layers(x) # 直接调用 encoder 的 layers

        # 输出形状: (batch_size, num_patches, feature_dim)
        out = encoder_output[:, 1:, :]

        return out

    def fine_tune(self, fine_tune=False, num_blocks_to_tune=4):
        """
        允许或禁止计算 ViT 编码器层的梯度。

        :param fine_tune: 是否允许微调？
        :param num_blocks_to_tune: 如果微调，指定最后多少个 Transformer 块参与训练。
        """
        # 首先冻结所有参数
        for param in self.vit.parameters():
            param.requires_grad = False

        if fine_tune:
            # 仅解冻最后几个 Transformer 块的参数
            if num_blocks_to_tune > 0 and hasattr(self.vit, 'encoder') and hasattr(self.vit.encoder, 'layers'):
                # ViT 的 Transformer 块存储在 self.vit.encoder.layers 中
                for layer in self.vit.encoder.layers[-num_blocks_to_tune:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"ViTEncoder Info: Fine-tuning enabled for the last {num_blocks_to_tune} transformer blocks.")
            else:
                 print(f"ViTEncoder Warning: Could not fine-tune last {num_blocks_to_tune} blocks. "
                       "Check model structure or num_blocks_to_tune value.")

            # (可选) 解冻其他可能需要微调的部分，例如 patch embedding 或最终的 LayerNorm
            # if hasattr(self.vit, 'conv_proj'):
            #     for param in self.vit.conv_proj.parameters():
            #         param.requires_grad = True
            # if hasattr(self.vit, 'ln'): # 如果使用了最终的 Layer Normalization
            #      for param in self.vit.ln.parameters():
            #          param.requires_grad = True

# --- 使用示例 (可选) ---
if __name__ == '__main__':
    encoder = ViTEncoder(encoded_image_size=14, model_name='vit_b_16', pretrained=True)
    encoder.eval() # 设置为评估模式

    # 创建一个符合 vit_b_16 期望输入的示例批次 (batch_size=4, channels=3, height=224, width=224)
    dummy_images = torch.randn(4, 3, 224, 224)

    # 前向传播
    encoded_output = encoder(dummy_images)

    # 打印输出形状 (应为 [4, 14, 14, 768])
    print("Input shape:", dummy_images.shape)
    print("Output shape:", encoded_output.shape)
    print("Feature dimension:", encoder.feature_dim)

    # 测试微调设置
    print("\nTesting fine-tuning:")
    encoder.fine_tune(fine_tune=True, num_blocks_to_tune=2)
    # 检查最后两个块的参数是否需要梯度
    for i, layer in enumerate(encoder.vit.encoder.layers):
        requires_grad_list = [p.requires_grad for p in layer.parameters()]
        all_require_grad = all(requires_grad_list)
        print(f"Block {i} requires_grad: {all_require_grad}")
