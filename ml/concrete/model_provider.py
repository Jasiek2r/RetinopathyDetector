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

class RetFoundViT(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        cls = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(cls)

class ModelProvider:
    def create_conv_model(self, num_classes=5):
        model = timm.create_model(
            "efficientnet_b0",
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

        hidden = backbone.config.hidden_size

        classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        return DinoRetinopathyModel(backbone, classifier)

    def create_retfound(self, num_classes=5):
        model_id = "iszt/RETFound_dinov2_meh"

        backbone = AutoModel.from_pretrained(model_id)

        hidden = backbone.config.hidden_size

        classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        return RetFoundViT(backbone, classifier)
