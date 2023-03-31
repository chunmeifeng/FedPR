#!/usr/bin/env python3

"""
ViT-related models
Note: models return logits instead of prob
"""
import torch.nn as nn
from models.vit_prompt.swin_transformer import SwinTransformer, PromptedSwinTransformer
from models.decode_heads import VisionTransformerUpHead


class Swin(nn.Module):
    """Swin-related model."""

    def __init__(self, cfg):
        super(Swin, self).__init__()

        if "prompt" in cfg.MODEL.TRANSFER_TYPE:
            prompt_cfg = cfg.MODEL.PROMPT
        else:
            prompt_cfg = None

        if cfg.MODEL.TRANSFER_TYPE != "end2end" and "prompt" not in cfg.MODEL.TRANSFER_TYPE:
            # linear, cls, tiny-tl, parital, adapter
            self.froze_enc = True
        else:
            # prompt, end2end, cls+prompt
            self.froze_enc = False

        self.build_backbone(prompt_cfg, cfg)
        self.cfg = cfg
        self.setup_head(cfg)

    def build_backbone(self, prompt_cfg, cfg):
        transfer_type = cfg.MODEL.TRANSFER_TYPE
        self.enc, self.feat_dim = build_swin_model(cfg.MODEL.SUBTYPE, cfg.MODEL.INPUTSIZE, prompt_cfg)

        # linear, prompt, cls, cls+prompt, partial_1
        if transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False


    def setup_head(self, cfg):
        self.head = VisionTransformerUpHead(img_size=cfg.MODEL.FINALSIZE, embed_dim=self.feat_dim, norm_cfg={'type': 'BN'},
                                num_conv=cfg.MODEL.HEAD_NUM_CONV)

    def forward(self, x, return_feature=False):

        if self.froze_enc and self.enc.training:
            self.enc.eval()
        x = self.enc(x)  # batch_size x self.feat_dim

        if return_feature:
            return x, x
        x = self.head(x)

        return x



def build_swin_model(model_type, crop_size, prompt_cfg):
    if prompt_cfg is not None:
        return _build_prompted_swin_model(
            model_type, crop_size, prompt_cfg)
    else:
        return _build_swin_model(model_type, crop_size)


def _build_prompted_swin_model(model_type, crop_size, prompt_cfg):
    if model_type == "swin_320":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            in_chans=1,
            patch_size=8,
            embed_dim=256,
            depths=[22],
            num_heads=[8],
            window_size=10,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 256
    feat_dim = int(embed_dim)


    return model, feat_dim


def _build_swin_model(model_type, crop_size):

    if model_type == "swinb_fastmri_320":
        model = SwinTransformer(
            img_size=crop_size,
            in_chans=1,
            patch_size=8,
            embed_dim=256,
            depths= [22],
            num_heads= [8],
            window_size=10,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 256
        feat_dim = int(embed_dim)

    return model, feat_dim