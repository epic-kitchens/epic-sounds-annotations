# -*- coding: utf-8 -*-
# @Time    : 7/16/21 3:12 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

# the unified ast models for all pretraining/fine-tuning tasks.
import torch.nn as nn
import numpy as np
import tempfile
import random
import torch
import timm

from timm.models.layers import trunc_normal_
from timm.models.layers import to_2tuple
from random import randrange

import slowfast.utils.logging as logging

from slowfast.config.defaults import get_cfg
from .build import MODEL_REGISTRY


logger = logging.get_logger(__name__)

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

@MODEL_REGISTRY.register()
class SSASTModel(nn.Module):
    def __init__(self, cfg):

        super(SSASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        # Override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        self.task = cfg.SSAST.TASK
        self.cluster = cfg.SSAST.CLUSTER
        self.mask_patch = cfg.SSAST.MASK_PATCH
        
        if cfg.SSAST.PRETRAIN_STAGE == True: # Pretrain the AST model
            self.init_pretrained(cfg)
        else:  # Use a pretrained model for finetuning
            self.init_finetuned(cfg)

    def init_pretrained(self, cfg):
        if cfg.SSAST.FSTRIDE != cfg.SSAST.FSHAPE or cfg.SSAST.TSTRIDE != cfg.SSAST.TSHAPE:
            raise ValueError('fstride != fshape or tstride != tshape, they must be same at the pretraining stage, patch split overlapping is not supported.')

        # If AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if cfg.SSAST.MODEL_SIZE == 'tiny':
            self.v = timm.create_model(
                    'vit_deit_tiny_distilled_patch16_224', 
                    pretrained=False
                )
            self.heads, self.depth = 3, 12
            self.cls_token_num = 2
        elif cfg.SSAST.MODEL_SIZE == 'small':
            self.v = timm.create_model(
                    'vit_deit_small_distilled_patch16_224', 
                    pretrained=False
                )
            self.heads, self.depth = 6, 12
            self.cls_token_num = 2
        elif cfg.SSAST.MODEL_SIZE == 'base':
            self.v = timm.create_model(
                    'vit_deit_base_distilled_patch16_384', 
                    pretrained=False
                )
            self.heads, self.depth = 12, 12
            self.cls_token_num = 2
        elif cfg.SSAST.MODEL_SIZE == 'base_nokd':
            self.v = timm.create_model(
                    'vit_deit_base_patch16_384', 
                    pretrained=False
                )
            self.heads, self.depth = 12, 12
            self.cls_token_num = 1
        else:
            raise Exception('Model size must be one of tiny, small, base, base_nokd')

        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]

        # SSL Pretraining Code
        self.softmax = nn.Softmax(dim=-1)
        self.lsoftmax = nn.LogSoftmax(dim=-1)
        self.fshape, self.tshape = cfg.SSAST.FSHAPE, cfg.SSAST.TSHAPE
        self.fstride, self.tstride = cfg.SSAST.FSTRIDE, cfg.SSAST.TSTRIDE
        self.input_fdim = cfg.AUDIO_DATA.NUM_FREQUENCIES 
        self.input_tdim = cfg.AUDIO_DATA.NUM_FRAMES
        # This is a trick to make state_dict to track pretraining input_fdim 
        # and input_tdim and save them by using torch.save
        self.p_input_fdim = nn.Parameter(
                                torch.tensor(cfg.AUDIO_DATA.NUM_FREQUENCIES), 
                                requires_grad=False
                            ) 
        self.p_input_tdim = nn.Parameter(
                                torch.tensor(cfg.AUDIO_DATA.NUM_FRAMES), 
                                requires_grad=False
                            )
        # Masked patch classification (discriminative objective) layer. We 
        # use two layers for pretext task, but using a single layer has 
        # similar performance. We map the output of transformer 
        # (768-dim for base models) to 256-dim patch input space, 
        # and then dot product with flattened patch input (also 256-dim) to 
        # calculate loss. Alternatively, you can map the output of 
        # transformer to 768-dim patch embedding space, and dot product with 
        # patch embedding. Performance-wise they are similar, but map to 256 
        # space is more efficient.
        self.cpredlayer = nn.Sequential(
                                nn.Linear(
                                        self.original_embedding_dim, 
                                        self.original_embedding_dim
                                    ), 
                                nn.ReLU(), 
                                nn.Linear(
                                        self.original_embedding_dim, 
                                        256
                                    )
                                )
        # Masked patch reconstruction (generative objective) layer
        self.gpredlayer = nn.Sequential(
                                nn.Linear(
                                        self.original_embedding_dim, 
                                        self.original_embedding_dim
                                    ), 
                                nn.ReLU(), 
                                nn.Linear(
                                        self.original_embedding_dim, 
                                        256
                                    )
                                )
        self.unfold = torch.nn.Unfold(
                        kernel_size=(cfg.SSAST.FSHAPE, cfg.SSAST.TSHAPE), 
                        stride=(cfg.SSAST.FSTRIDE, cfg.SSAST.TSTRIDE)
                        )

        # We use learnable mask embedding (follow the BEIT paper), 
        # but using a fixed mask embedding (e.g., 0) leads to same 
        # performance.
        self.mask_embed = nn.Parameter(
                            torch.zeros([1, 1, self.original_embedding_dim])
                        )
        self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

        # Get the intermediate shape
        self.p_f_dim, self.p_t_dim = self.get_shape(
                                        cfg.SSAST.FSTRIDE, 
                                        cfg.SSAST.TSTRIDE, 
                                        cfg.AUDIO_DATA.NUM_FREQUENCIES, 
                                        cfg.AUDIO_DATA.NUM_FRAMES, 
                                        cfg.SSAST.FSHAPE, 
                                        cfg.SSAST.TSHAPE
                                    )
        num_patches = self.p_f_dim * self.p_t_dim
        self.num_patches = num_patches
        self.v.patch_embed.num_patches = num_patches
        logger.info('Pretraining patch split stride: frequency={:d}, time={:d}'.format(cfg.SSAST.FSTRIDE, cfg.SSAST.TSTRIDE))
        logger.info('Pretraining patch shape: frequency={:d}, time={:d}'.format(cfg.SSAST.FSHAPE, cfg.SSAST.TSHAPE))
        logger.info('Pretraining patch array dimension: frequency={:d}, time={:d}'.format(self.p_f_dim, self.p_t_dim))
        logger.info('Pretraining number of patches={:d}'.format(num_patches))

        # The linear patch projection layer, use 1 channel for spectrogram 
        # rather than the original 3 channels for RGB images.
        new_proj = torch.nn.Conv2d(
                        1, 
                        self.original_embedding_dim, 
                        kernel_size=(cfg.SSAST.FSHAPE, cfg.SSAST.TSHAPE), 
                        stride=(cfg.SSAST.FSTRIDE, cfg.SSAST.TSTRIDE)
                    )
        self.v.patch_embed.proj = new_proj

        # Use trainable positional embedding
        new_pos_embed = nn.Parameter(
                            torch.zeros(
                                1, 
                                self.v.patch_embed.num_patches + self.cls_token_num, 
                                self.original_embedding_dim
                            )
                        )
        self.v.pos_embed = new_pos_embed
        trunc_normal_(self.v.pos_embed, std=.02)

    def init_finetuned(self, cfg):
        device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
        model_path = cfg.TRAIN.CHECKPOINT_FILE_PATH if cfg.TRAIN.ENABLE else cfg.TEST.CHECKPOINT_FILE_PATH

        sd = torch.load(model_path, map_location=device)
        sd = sd["model_state"] if "model_state" in sd.keys() else sd
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        
        # Get the fshape and tshape, input_fdim and input_tdim in the pretraining stage
        if 'v.patch_embed.proj.weight' in sd.keys():
            p_fshape = sd['v.patch_embed.proj.weight'].shape[2]
            p_tshape = sd['v.patch_embed.proj.weight'].shape[3]
        else:
            raise ValueError(f'Error loading patch embedding from {model_path}')

        if "p_input_fdim" in sd.keys() and "p_input_tdim" in sd.keys():
            self.p_input_fdim = nn.Parameter(
                            sd['p_input_fdim'],
                            requires_grad=False
                        ) # Wrap in nn.Parameter to save to state dict
            self.p_input_tdim = nn.Parameter(
                            sd['p_input_tdim'],
                            requires_grad=False
                        )
        else:
            raise ValueError(f'Error loading pretrained input fdim/tdim from {model_path}')


        logger.info(f'Initialising SSL pretrained model parameters from {model_path}')
        # During pretraining, fstride=fshape and tstride=tshape because no 
        # patch overlapping is used. Here, input_fdim and input_tdim should 
        # be equal to that used in pretraining, not that in the fine-tuning.
        # We need to know input_fdim and input_tdim to do positional 
        # embedding cut/interpolation. Generally it should be better to use 
        # same input_fdim during pretraining and finetuning, but input_tdim 
        # can be safely different
        p_cfg = cfg
        p_cfg.SSAST.FSTRIDE = p_fshape
        p_cfg.SSAST.TSTRIDE = p_tshape
        p_cfg.SSAST.FSHAPE = p_fshape
        p_cfg.SSAST.TSHAPE = p_tshape
        p_cfg.AUDIO_DATA.NUM_FREQUENCIES = self.p_input_fdim.item()
        p_cfg.AUDIO_DATA.NUM_FRAMES = self.p_input_tdim.item()
        p_cfg.SSAST.PRETRAIN_STAGE = True

        audio_model = SSASTModel(p_cfg)
        self.v = audio_model.v 
        self.original_embedding_dim = self.v.pos_embed.shape[2] 
        self.cls_token_num = audio_model.cls_token_num 

        # MLP head for fine-tuning
        self.mlp_head = nn.Sequential(
                                nn.LayerNorm(
                                        self.original_embedding_dim
                                    ),
                                nn.Linear(
                                        self.original_embedding_dim, 
                                        cfg.MODEL.NUM_CLASSES[0]
                                    )
                            )   
        f_dim, t_dim = self.get_shape(
                                cfg.SSAST.FSTRIDE, 
                                cfg.SSAST.TSTRIDE, 
                                cfg.AUDIO_DATA.NUM_FREQUENCIES, 
                                cfg.AUDIO_DATA.NUM_FRAMES, 
                                cfg.SSAST.FSHAPE, 
                                cfg.SSAST.TSHAPE
                            )
        # Patch array dimension during pretraining
        p_f_dim, p_t_dim = audio_model.p_f_dim, audio_model.p_t_dim
        num_patches = f_dim * t_dim
        p_num_patches = p_f_dim * p_t_dim
        self.v.patch_embed.num_patches = num_patches
        logger.info('Fine-tuning patch split stride: frequncey={:d}, time={:d}'.format(cfg.SSAST.FSTRIDE, cfg.SSAST.TSTRIDE))
        logger.info('Fine-tuning number of patches={:d}'.format(num_patches))

        # Patch shape should be same for pretraining and fine-tuning
        if cfg.SSAST.FSHAPE != p_fshape or cfg.SSAST.TSHAPE != p_tshape:
            raise ValueError('The patch shape of pretraining and fine-tuning is not consistant, pretraining: f={:d}, t={:d}, finetuning: f={:d}, t={:d}'.format(p_fshape, p_tshape, cfg.SSAST.FSHAPE, cfg.SSAST.TSHAPE))

        # Patch split stride generally should be different for pretraining 
        # and fine-tuning, as patch split overlapping is only used in 
        # finetuning. During pretraining, p_fshape = p_fstride and 
        # p_tshape = p_tstride
        if cfg.SSAST.FSTRIDE != p_fshape or cfg.SSAST.TSTRIDE != p_tshape:
            # Initialize a new patch embedding layer with desired new stride
            new_proj = torch.nn.Conv2d(
                            1, 
                            self.original_embedding_dim, 
                            kernel_size=(cfg.SSAST.FSHAPE, cfg.SSAST.TSHAPE), 
                            stride=(cfg.SSAST.FSTRIDE, cfg.SSAST.TSTRIDE)
                        )
            # But the weights of patch embedding layer is still got 
            # from the pretrained models
            new_proj.weight = torch.nn.Parameter(
                                torch.sum(
                                    self.v.patch_embed.proj.weight, 
                                    dim=1
                                ).unsqueeze(1)
                            )
            new_proj.bias = self.v.patch_embed.proj.bias

            self.v.patch_embed.proj = new_proj

        # If fine-tuning, initialise new pos embedding
        new_pos_embed = self.v.pos_embed[
                            :, 
                            self.cls_token_num:, 
                            :
                        ].detach().reshape(
                            1, 
                            p_num_patches, 
                            self.original_embedding_dim
                        ).transpose(1, 2).reshape(
                            1, 
                            self.original_embedding_dim, 
                            p_f_dim, 
                            p_t_dim
                        )
        # Cut or interpolate the positional embedding
        if t_dim < p_t_dim:
            new_pos_embed = new_pos_embed[
                :, 
                :, 
                :, 
                int(p_t_dim / 2) - int(t_dim / 2): int(p_t_dim / 2) + int(t_dim / 2) + t_dim
            ]
        else:
            new_pos_embed = torch.nn.functional.interpolate(
                                new_pos_embed, 
                                size=(8, t_dim), 
                                mode='bilinear'
                            )
        if f_dim < p_f_dim:
            new_pos_embed = new_pos_embed[
                :, 
                :, 
                int(p_f_dim / 2) - int(f_dim / 2): int(p_f_dim / 2) - int(f_dim / 2) + t_dim, 
                :
            ]
        else:
            new_pos_embed = torch.nn.functional.interpolate(
                                new_pos_embed, 
                                size=(f_dim, t_dim), 
                                mode='bilinear'
                            )

        new_pos_embed = new_pos_embed.reshape(
                                        1, 
                                        self.original_embedding_dim, 
                                        num_patches
                                    ).transpose(1, 2)
        self.v.pos_embed = nn.Parameter(
            torch.cat(
                [
                    self.v.pos_embed[:, :self.cls_token_num, :].detach(), 
                    new_pos_embed
                ], 
                dim=1
            )
        )


    # get the shape of intermediate representation.
    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    # Generate mask for 16*16 patch
    def gen_maskid_patch(self, sequence_len=512, mask_size=100, cluster=3):
        mask_id = []

        # Randomize clutering factor in [3,6)
        cur_clus = randrange(cluster) + 3

        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)

            ## This improves the efficiency, but might change the pretrained model
            # while start_id in mask_id:
            #     start_id = randrange(sequence_len)

            cur_mask = []
            for i in range(0, cur_clus):
                for j in range(0, cur_clus):
                    mask_cand = start_id + self.p_t_dim * i + j
                    if mask_cand > 0 and mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)

    # Using cluster for frame masking hurts the performance, so just use the 
    # naive random sampling
    def gen_maskid_frame(self, sequence_len=512, mask_size=100):
        mask_id = random.sample(range(0, sequence_len), mask_size)
        return torch.tensor(mask_id)

    def finetuningavgtok(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)

        # Average output of all tokens except cls token(s)
        x = torch.mean(x[:, self.cls_token_num:, :], dim=1)
        return x

    def finetuningcls(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)

        # If models has two cls tokens (DEIT), average as the clip-level 
        # representation
        if self.cls_token_num == 2:
            x = (x[:, 0] + x[:, 1]) / 2
        else:
            x = x[:, 0]
   
        return self.mlp_head(x)

    # Masked patch pretraining with discriminative objective
    def mpc(self, x, mask_patch, cluster, show_mask=False):
        input = self.unfold(x).transpose(1, 2)
        B = x.shape[0]
        # x in shape (batch_size, sequence_len, embedding dim)
        x = self.v.patch_embed(x)

        # Encode the patch
        # size 12(batch_size) * 100(#mask_patch) * 768(hidden_dim), 
        # prepare to save the true values of masked samples
        encode_samples = torch.empty((B, mask_patch, 256), device=x.device, requires_grad=False).float()
        # Size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # Size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)

        # For each audio clip in the batch
        for i in range(B):
            # Randomly generate #mask_patch mask indexes without duplicate
            if cluster == True:
                # Use this if you are masking e.g. 16*16 patches
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                # Use this if you are masking frame, i.e., 128*2 patches
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            # Copy the masked embeddings, note gradients are stopped in this path
            encode_samples[i] = input[i, mask_index[i], :].clone().detach()
            # Mask the encode samples with 0
            mask_dense[i, mask_index[i], :] = 0

        # Follow BEIT paper, mask with learnable masking embedding, 
        # but no performance diff observed compared with masking with 0s.
        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # Mask the patch
        x = x * mask_dense + (1-mask_dense) * mask_tokens

        # Pass through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        # Prediction of the masked patch
        pred = torch.empty((B, mask_patch, 256), device=x.device).float()  # e.g. size 12*100*768
        for i in range(B):
            #  +2 for indexes because skipping the cls and dis token.
            # We map the output of transformer (768-dim for base models) to 
            # 256-dim patch input space, and then dot product with flattened 
            # patch input (also 256-dim) to calculate loss. Alternatively, you 
            # can map the output of transformer to 768-dim patch embedding 
            # space, and dot product with patch embedding. Performance-wise they 
            # are similar, but map to 256 space is more efficient.
            pred[i] = self.cpredlayer(x[i, mask_index[i] + self.cls_token_num, :])

        # Calculate the NCE loss
        nce = torch.tensor(0.0).to(x.device)
        correct = torch.tensor(0.0).to(x.device)
        for i in np.arange(0, B):
            # Negative samples are from the same batch.
            # 8/12/2022: has a difference with equation (1) in the ssast paper 
            # but (likely) performance-wise similar, 
            # see https://github.com/YuanGongND/ssast/issues/13
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 100*100
            correct += torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, mask_patch, device=x.device)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        acc = 1. * correct / (B * mask_patch)
        nce = nce / (-1. * B * mask_patch)

        # Visualize the masked area, for probing test only, set 
        # show_mask = False for any training/inference.
        if show_mask == False:
            return acc, nce
        else:
            if B > 1:
                raise Exception('Currently only support single spectrogram probing test.')

            self.mask_correct = torch.nn.Parameter(torch.arange(0, mask_patch), requires_grad=False)

            pred = input.clone()  # [B, 512, 256]
            masked = input.clone()

            for i in range(B):
                result = [float(t) * 99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)]
                pred[i, mask_index[i], :] = torch.tensor(result).reshape(mask_patch, 1).expand(mask_patch, 256)
                masked[i, mask_index[i], :] = 99.0

            fold = torch.nn.Fold(output_size=([self.input_fdim, self.input_tdim]), kernel_size=(self.fshape, self.tshape), stride=(self.fstride, self.tstride))
            pred = fold(pred.transpose(1, 2))
            masked = fold(masked.transpose(1, 2))

            return pred, masked

    # Masked patch pretraining with generative objective
    def mpg(self, input, mask_patch, cluster):
        B = input.shape[0]
        x = self.v.patch_embed(input)
        input = self.unfold(input).transpose(1, 2)

        # Size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # Size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        for i in range(B):
            # Randomly generate #mask_patch mask indexes without duplicate
            if cluster == True:
                # Use this if you are masking e.g. 16*16 patches
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                # Use this if you are masking frame, i.e., 128*2 patches
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            mask_dense[i, mask_index[i], :] = 0

        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # Follow BEIT paper, mask with learnable masking embedding, 
        # but no performance diff observed compared with masking with 0s.
        x = x * mask_dense + (1-mask_dense) * mask_tokens

        # Go through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        pred = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float()  # e.g. size 12*100*256
        target = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float() # e.g. size 12*100*256

        for i in range(B):
            #  +2 for indexes because cls and dis token
            pred[i] = self.gpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
            target[i] = input[i, mask_index[i], :]

        # Calculate the MSE loss
        mse = torch.mean((pred - target) ** 2)

        return mse

    def forward(self, x, feature_extraction=False):
        # Expect input x = (B, C, T, F)
        x = x[0].transpose(2, 3)

        # Finetuning (ft), use the mean of all token (patch) output as 
        # clip-level representation. This is default for SSAST fine-tuning as 
        # during pretraining, supervision signal is given to each token, 
        # not the [cls] token
        if self.task == 'ft_avgtok':
            return self.finetuningavgtok(x)
        # Alternatively, use the [cls] token output as clip-level representation.
        elif self.task == 'ft_cls':
            return self.finetuningcls(x)
        # Pretraining, masked patch classification (discriminative objective)
        elif self.task == 'pretrain_mpc':
            return self.mpc(x, mask_patch=self.mask_patch, cluster=self.cluster)
        # Pretraining, masked patch reconstruction (generative objective)
        elif self.task == 'pretrain_mpg':
            return self.mpg(x, mask_patch=self.mask_patch, cluster=self.cluster)
        elif self.task == 'visualize_mask':
            return self.mpc(x, mask_patch=self.mask_patch, cluster=self.cluster, show_mask=True)
        else:
            raise Exception('Task unrecognized.')

    def freeze_fn(self, freeze_mode):
        if freeze_mode == 'freeze_backbone':
            logger.info("Freezing all backbone layers")
            for n, p in self.named_parameters():
                if 'mlp_head' not in n and "v.pos_embed" not in n and "cls_token" not in n:
                    # shutdown parameters update in frozen mode
                    p.requires_grad_(False)
