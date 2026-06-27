import timm
from torch import nn
from transformers import AutoModel


class DinoRetinopathyModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)


class ModelProvider:
    def create_conv_model(self, num_classes=5):
        model = timm.create_model(
            "convnext_small",
            pretrained=True,
            num_classes=num_classes
        )

        return model

    # -------------------------
    # MODEL
    # -------------------------
    def create_model(self, num_classes=5):
        backbone = AutoModel.from_pretrained("facebook/dinov2-base")

        # freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False
        for name, param in backbone.named_parameters():
            if "blocks.11" in name or "blocks.10" in name:
                param.requires_grad = True

        hidden = backbone.config.hidden_size

        classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        return DinoRetinopathyModel(backbone, classifier)
