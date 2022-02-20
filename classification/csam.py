from __future__ import print_function
import argparse
from calendar import c
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pdb
from tqdm import tqdm
from positional_encodings import PositionalEncoding2D
"""
Modified from: https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

class ConvolutionalSelfAttention(nn.Module):
    def __init__(self,
        spatial_shape,
        filter_size,
        approach_args={'approach_name': '4', 'padding': 'valid', 'stride': 1}
    ):
        super(ConvolutionalSelfAttention, self).__init__()

        self.approach_args = approach_args
        self.approach_name = approach_args.APPROACH_NAME # ['approach_name']
        self.padding_type = approach_args.PADDING #['padding']
        self.apply_stochastic_stride = approach_args.APPLY_STOCHASTIC_STRIDE#['apply_stochastic_stride']
        self.stride = approach_args.STRIDE#['stride']
        self.softmax_temp = approach_args.SOFTMAX_TEMP#['softmax_temp']
        self.use_residual_connection = approach_args.USE_RESIDUAL_CONNECTION#['use_residual_connection']

        self.filter_K = filter_size
        self.filter_size = self.filter_K * self.filter_K

        self.input_H, self.input_W, self.spatial_C = spatial_shape
        self.input_padder = self.compute_padding(self.padding_type)
        self.spatial_H = int(self.input_H + 2 * self.padding_tuple[2])
        self.spatial_W = int(self.input_W + 2 * self.padding_tuple[0])
        if 'self_attention' in self.approach_name or self.approach_args.POS_EMB_DIM == -1:
            self.pos_emb_dim = self.spatial_C
        else:
            self.pos_emb_dim =  self.approach_args.POS_EMB_DIM

        h_convs, w_convs = self.get_num_convs()
        output_shrunk = h_convs < self.input_H or w_convs < self.input_W
        if self.use_residual_connection and output_shrunk:
            self.upsampler = torch.nn.Upsample(size=(self.input_H, self.input_W), mode='bilinear')
        else:
            self.upsampler = torch.nn.Identity()

        self.setup_approach()
        self.name2approach = {
            '1': self.efficient_approach1,
            'Two': self.efficient_approach2,
            'Three': self.forward_on_approach3,
            '4': self.forward_on_approach4,
            '5': self.forward_on_approach5,
            '4_mem_efficient': self.forward_on_approach4_mem_efficient,
            'self_attention': self.forward_on_self_attention,
            'reversed_self_attention_elemwise': self.forward_on_reverse_self_attention,
            'reversed_self_attention_normal': self.forward_on_reverse_self_attention,
            'background_self_attention': self.forward_on_background_self_attention,
        }

        self.local_mask = self.compute_input_mask()
        if self.approach_name in {'Three', '4', '4_mem_efficient'}:
            new_shape = [self.num_convs, 1, self.spatial_H, self.spatial_W, 1]                                      # [Nc,1,H,W,1]
            self.local_mask = self.local_mask.reshape(*new_shape)
            self.padding_mask = self.padding_mask.reshape([1, 1, self.spatial_H, self.spatial_W, 1])

        if torch.cuda.is_available():
            self.local_mask = self.local_mask.cuda()
            self.local_indices = self.local_indices.cuda()
            self.padding_mask = self.padding_mask.cuda()

    def compute_padding(self, padding_type):
        if padding_type.lower() == 'valid' or self.approach_name in {
            'self_attention', 'reversed_self_attention', 
            'reversed_self_attention_elemwise', 'reversed_self_attention_normal',
            }:
            padding_tuple = (0, 0, 0, 0)
        elif padding_type.lower() == 'same':
            padding_x = int((self.stride * (self.input_W - 1) - self.input_W + self.filter_K) / 2)
            padding_y = int((self.stride * (self.input_H - 1) - self.input_H + self.filter_K) / 2)
            padding_tuple = (padding_x, padding_x, padding_y, padding_y)
        else:
            raise ValueError('Unknown padding type requested')
        self.padding_tuple = padding_tuple
        input_padder = nn.ConstantPad2d(padding_tuple, 0.)
        # Compute padding indicator mask
        padding_mask = torch.ones((self.input_H, self.input_W))
        self.padding_mask = 1 - input_padder(padding_mask.unsqueeze(0)).squeeze(0)
        return input_padder

    # expects image to be [B,C,H,W]
    def undo_padding(self, image):
        _, _, H, W = image.shape
        left_pad, right_pad, top_pad, bottom_pad = self.padding_tuple
        return image[:, :, top_pad:H-bottom_pad, left_pad:W-right_pad]

    def get_num_convs(self):
        h_dim = int((self.spatial_H - self.filter_K) / self.stride) + 1
        w_dim = int((self.spatial_W - self.filter_K) / self.stride) + 1
        return h_dim, w_dim

    def get_output_shape(self):
        if self.use_residual_connection:
            h_dim, w_dim = self.input_H, self.input_W
        else:
            h_dim, w_dim = self.get_num_convs()
        return [h_dim, w_dim, self.spatial_C]

    def setup_approach(self):
        self.X_encoding_dim = self.spatial_C                                                                            # Call this E
        # Account for positional encodings
        self.maybe_create_positional_encodings()
        if self.pos_emb_dim > 0:
            self.X_encoding_dim += self.pos_emb_dim
        if self.approach_name == '1':
            self.global_transform = nn.Linear(self.X_encoding_dim, self.filter_K * self.filter_K)
        elif self.approach_name == 'Two':
            self.global_transform = nn.Linear(self.X_encoding_dim, 1)
        elif self.approach_name == 'Three':
            self.global_transform = nn.Linear(self.X_encoding_dim, self.spatial_C)
        elif self.approach_name == '4' or self.approach_name == '4_mem_efficient':
            self.key_transform = nn.Linear(self.X_encoding_dim, self.X_encoding_dim)
            self.query_transform = nn.Linear(self.X_encoding_dim, self.X_encoding_dim)
            self.value_transform = nn.Linear(self.X_encoding_dim, 1)
        elif self.approach_name == '5':
            self.key_transform = nn.Linear(self.X_encoding_dim, self.spatial_C)
            self.query_transform = nn.Linear(self.X_encoding_dim, self.spatial_C)
            self.value_transform = nn.Linear(self.X_encoding_dim, self.spatial_C)
        elif self.approach_name in {
            'self_attention', 'reversed_self_attention', 
            'reversed_self_attention_elemwise', 'reversed_self_attention_normal',
            'background_self_attention'
            }:
            self.key_transform = nn.Linear(self.spatial_C, self.spatial_C)
            self.query_transform = nn.Linear(self.spatial_C, self.spatial_C)
            self.value_transform = nn.Linear(self.spatial_C, self.spatial_C)
            if self.approach_name == 'background_self_attention':
                self.elem_scales = nn.Parameter(torch.rand(self.spatial_H, self.spatial_W, 1, requires_grad=True))
        else:
            raise ValueError('Invalid Approach type')

    def maybe_create_positional_encodings(self):
        if self.pos_emb_dim > 0:
            self.positional_encodings = PositionalEncoding2D(self.pos_emb_dim)
        else:
            self.positional_encodings = None

    def masked_softmax(self, vec, mask, dim=1, epsilon=1e-5):
        # pdb.set_trace()
        # Adapted from: https://discuss.pytorch.org/t/apply-mask-softmax/14212/14
        vec = vec * mask
        exps = torch.exp(vec - vec.max(dim, keepdim=True)[0])
        masked_exps = exps * mask
        masked_sums = masked_exps.sum(dim, keepdim=True) #+ epsilon
        return (masked_exps/masked_sums)

    def cudaify_module(self):
        self.local_mask = self.local_mask.cuda()
        self.local_indices = self.local_indices()

    def compute_cell_lengths(self, num_cells, total_length):
        cell_lengths = self.stride * torch.ones(num_cells, dtype=torch.int)

        # convs_height and convs_width are not necessarily divisible by stride
        if self.apply_stochastic_stride:
            leftover_length = total_length - self.stride * num_cells
            for _ in range(leftover_length):
                i = random.randrange(len(cell_lengths))
                cell_lengths[i] += 1

        return cell_lengths

    def compute_input_mask(self):
        convs_height = self.spatial_H - self.filter_K + 1
        convs_width = self.spatial_W - self.filter_K + 1
        num_convs = convs_height * convs_width

        strided_convs_height = int(convs_height // self.stride)
        strided_convs_width = int(convs_width // self.stride)
        num_strided_convs = int(strided_convs_height * strided_convs_width)

        cell_heights = self.compute_cell_lengths(strided_convs_height, convs_height)
        cell_widths = self.compute_cell_lengths(strided_convs_width, convs_width)

        cell_row_starts = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.int), cell_heights[:-1])), dim=0)
        cell_col_starts = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.int), cell_widths[:-1])), dim=0)

        input_mask = torch.zeros(
            (
                num_strided_convs,
                self.spatial_H,
                self.spatial_W,
            ),
            dtype=torch.float32)

        self.local_indices = torch.zeros((num_strided_convs, self.filter_K * self.filter_K))
        conv_idx = 0
        for i_idx, i in enumerate(cell_row_starts):
            for j_idx, j in enumerate(cell_col_starts):
                offset_i = i + (random.randrange(cell_heights[i_idx]) if self.apply_stochastic_stride else 0)
                offset_j = j + (random.randrange(cell_widths[j_idx]) if self.apply_stochastic_stride else 0)
                input_mask[conv_idx, offset_i:offset_i+self.filter_K, offset_j:offset_j+self.filter_K] = 1.
                self.local_indices[conv_idx] = input_mask[conv_idx].reshape(-1).nonzero().sort()[0].reshape(-1)
                input_mask[conv_idx] = F.relu(input_mask[conv_idx] - self.padding_mask)
                conv_idx += 1
        self.num_convs = strided_convs_height * strided_convs_width
        self.convs_height = strided_convs_height
        self.convs_width = strided_convs_width
        return input_mask.flatten(1)                                                                                    # [Nc, HW]

    def split_input(self, batch, mask_X_g=True):
        batch_r = batch.unsqueeze(0)                                                                                    # [1,B,H,W,E]
        if mask_X_g:
            X_g = batch_r * (1 - self.local_mask)                                                                       # [F,B,H,W,E]

            X_g = X_g.reshape(self.num_convs, -1, self.spatial_H * self.spatial_W, self.X_encoding_dim)                 # [F,B,HW,E]
            X_gi = torch.argsort(
                (X_g.sum(-1) != 0).type(torch.float32),
                dim=-1
            )[:, :, self.filter_size:].sort(dim=-1)[0]                                                                  # [F,B,(HW-K^2)]
            X_g_flat = X_g.reshape(-1, X_g.shape[2], X_g.shape[3])                                                      # [FB,HW,E]
            X_gi_flat = X_gi.reshape(-1, X_gi.shape[2])                                                                 # [FB,(HW-K^2)]
            X_g_flat_ = X_g_flat[torch.arange(X_g_flat.shape[0]).unsqueeze(-1), X_gi_flat]                              # [FB,(HW-K^2),E]
            X_g_ = X_g_flat_.reshape(
                self.num_convs, -1, X_g_flat_.shape[1], X_g_flat_.shape[2]
            )
        else:
            X_g_ = batch_r.reshape(1, -1, self.spatial_H * self.spatial_W, self.X_encoding_dim)                         # [1,B,HW,E]
        X_l = (batch_r * self.local_mask).reshape(                                                                      # [F,B,H,W,E]
            self.num_convs, -1, self.spatial_H * self.spatial_W, self.X_encoding_dim                                    # [F,B,HW,E]
        )
        X_li = torch.argsort(
            (X_l.sum(-1) != 0).type(torch.float32),
            dim=-1
        )[:, :, -self.filter_size:].sort(dim=-1)[0]                                                                     # [F,B,K^2]
        X_li_flat = X_li.reshape(-1, self.filter_size)                                                                  # [FB,K^2]
        X_l_flat = X_l.reshape(-1, X_l.shape[2], X_l.shape[3])                                                          # [FB,HW,E]
        X_l_flat_ = X_l_flat[torch.arange(X_l_flat.shape[0]).unsqueeze(-1), X_li_flat]                                  # [FB,K^2,E]
        X_l_ = X_l_flat_.reshape(self.num_convs, -1, self.filter_size, self.X_encoding_dim)                             # [F,B,K^2,E]
        return X_l_, X_g_

    def cosine_similarity(self, x, y):
        # Assume x, y are [F,B,*,E]
        x_normed = torch.nn.functional.normalize(x, dim=-1)
        y_normed = torch.nn.functional.normalize(y, dim=-1)
        if len(x.shape) == 3:
            y_permuted = y_normed.transpose(2, 1)
            res = torch.bmm(x_normed, y_permuted)
        else:
            y_permuted = y_normed.transpose(3, 2)
            res = torch.einsum('ijkl,ijla->ijka', x_normed, y_permuted)
        return res

    def efficient_approach1(self, batch):
        batch = self.maybe_add_positional_encodings(batch)
        batch_flat = batch.flatten(1,2)                                                                                 # [B,H,W,C] -> [B,HW,C]
        gX = self.global_transform(batch_flat)                                                                          # [B,HW,C] -> [B,HW,K^2]
        global_mask = (1 - self.padding_mask.reshape(1, -1)) - self.local_mask                                          # [Nc,HW]
        convolutions = torch.einsum(
            'ijk,kl->ijl',
            gX.transpose(2, 1),                                                                                         # [B,HW,K^2] -> [B,K^2,HW]
            global_mask.transpose(1,0)                                                                                  # [Nc,HW] -> [HW,Nc]
            ).transpose(2,1)                                                                                            # [B,K^2,HW]x[HW,Nc] -> [B,K^2,Nc] -> [B,Nc,K^2]
        local_idxs_onehot = F.one_hot(self.local_indices.type(torch.int64))                                             # OneHot([Nc,HW]) -> [Nc,K^2,HW]
        Xl = torch.einsum('ijk,lmj->ilmk', batch_flat, local_idxs_onehot.type(torch.float))                             # [B,HW,C],[Nc,K^2,HW] -> [B,Nc,K^2,C]
        result = torch.einsum('abcd,abc->abd', Xl, convolutions)                                                        # [B,Nc,K^2,C]x[B,Nc,K^2] -> [B,Nc,C]
        return result[:,:,:self.spatial_C].reshape(
            -1, self.convs_height, self.convs_width, self.spatial_C
            )                                                                                                           # [B,F,F,C]

    def maybe_add_positional_encodings(self, batch):
        if self.positional_encodings is not None:
            positional_encodings = self.positional_encodings(
                batch[:, :, :, :self.pos_emb_dim]                                                                       # [B,H,W,P]
            )
            if 'self_attention' in self.approach_name:
                return batch + positional_encodings
            return torch.cat((batch, positional_encodings), dim=-1)                                                     # [[B,H,W,C];[B,H,W,P]] -> [B,H,W,C+P]
        return batch

    def efficient_approach2(self, batch):

        batch_pos = self.maybe_add_positional_encodings(batch)
        global_mask = (1 - self.padding_mask.reshape(1, -1)) - self.local_mask                                          # [Nc,HW]
        batch_flat = batch_pos.flatten(1, 2)                                                                            # [B,H,W,C] -> [B,HW,C]
        gX = self.global_transform(batch_flat)                                                                          # [B,HW,C] -> [B,HW,1]
        cos_sim = self.cosine_similarity(batch_flat, batch_flat)                                                        # [B,HW,C]x[B,HW,C] -> [B,HW,HW]
        exp_sim = torch.exp(cos_sim - cos_sim.mean(dim=-1, keepdim=True)[0])                                            # [B,HW,HW]
        exp_sum = torch.einsum('ijk,kl->ijl', exp_sim, global_mask.transpose(1, 0))                                     # [B,HW,HW]x[HW,Nc] -> [B,HW,Nc]
        inverted_sum = 1. / exp_sum                                                                                     # [B,HW,Nc]
        masked_denominator = inverted_sum * self.local_mask.transpose(1,0).unsqueeze(0)                                 # [B,HW,Nc]x([Nc,Hw] -> [HW,Nc] -> [1,HW,Nc]) -> [B,HW,Nc]
        g_sim = gX.transpose(2, 1) * exp_sim                                                                            # ([B,HW,1] -> [B,1,HW])x[B,HW,HW] -> [B,HW,HW]
        g_sum = torch.einsum('ijk,kl->ijl', g_sim, global_mask.transpose(1, 0))                                         # [B,HW,HW]x([B,Nc,HW] -> [B,HW,Nc]) -> [B,HW,Nc]
        extended_filter = g_sum * masked_denominator                                                                    # [B,HW,Nc]
        result = torch.bmm(batch_flat.transpose(2, 1), extended_filter).transpose(2, 1)                                 # ([B,HW,C] -> [B,C,HW])x[B,HW,Nc] -> [B,C,Nc] -> [B,Nc,C]

        return result[:,:,:self.spatial_C].reshape(
            -1, self.convs_height, self.convs_width, self.spatial_C
        )                                                                                                               # [B,Nc,C] -> [B,F,F,C]

    def flops(self, N):
        if self.approach_name == 'Three':
            flops = 0
            # Apply Global Transform
            flops += self.spatial_H * self.spatial_W * self.spatial_C * self.spatial_C
            # Apply Window Attention
            # Input Normalization
            window_flops = self.spatial_H * self.spatial_W * self.spatial_C * 2
            # Similarity Creation
            window_flops += N * self.spatial_W * self.spatial_H * self.spatial_C * self.spatial_C
            # Softmax
            window_flops += N * self.spatial_W * self.spatial_H * 2
            # Filter Creation
            window_flops += N * self.spatial_H * self.spatial_H * self.spatial_H * self.spatial_W
            # Sum Projections
            window_flops += N * self.spatial_H * self.spatial_W
            # Apply Forget Gate & Sum & Sigmoid
            window_flops += N * self.spatial_C * 3
            # Apply Filter & Sum
            window_flops += N * self.spatial_C * 3
            flops += self.num_convs * window_flops
        
        elif self.approach_name == 'Two':
            flops = 0
            # Apply Global Transform
            flops += self.spatial_H * self.spatial_W * self.spatial_C
            # Cosine Similarity
            flops += (self.spatial_H * self.spatial_W * self.spatial_C * 2) + \
                (self.spatial_H * self.spatial_W * self.spatial_C * self.spatial_W * self.spatial_H)
            # Global Mask Applicator
            flops += self.spatial_C * self.num_convs * self.spatial_H * self.spatial_W
            # Inverse & Mask
            flops += self.spatial_W * self.spatial_H * self.spatial_C * 2
            # Apply Filter
            flops += self.spatial_H * self.spatial_W * self.spatial_C
            # Second Global Mask Applicator
            flops += self.spatial_C * self.num_convs * self.spatial_H * self.spatial_W
            # Finish product
            flops += self.spatial_H * self.spatial_W * self.spatial_C * self.num_convs

        return flops


    def forward_on_approach3(self, batch):
        global_mask = (1 - self.padding_mask - self.local_mask).flatten(
            start_dim=1, end_dim=-1).reshape(
                self.convs_height, self.convs_width, -1)                                                                # [Nc,1,H,W,1]
        X = self.maybe_add_positional_encodings(batch)                                                                  # [B,H,W,E]
        batch_size, H, W, _ = X.shape
        X_flat_spatial = X.view(-1, H * W, X.shape[-1])                                                                 # [B,HW,E]
        X_g_vectors = self.global_transform(X_flat_spatial)                                                             # [B,HW,C]

        convs_height = self.convs_height
        convs_width = self.convs_width
        # For cosine similarity
        X_normed = torch.nn.functional.normalize(X, dim=-1)

        output = torch.zeros(batch_size, convs_height, convs_width, self.spatial_C, dtype=torch.float).cuda()           # [B,F,F,C]
        for i in range(0, convs_height, self.stride):
            for j in range(0, convs_width, self.stride):
                X_l = X_normed[:, i:i+self.filter_K, j:j+self.filter_K]                                                 # [B,K,K,C]
                raw_compatibilities = torch.einsum('bhwc,bkjc->bhwkj', X_normed, X_l)                                   # [B,H,W,K,K]
                raw_compatibilities[:, i:i+self.filter_K, j:j+self.filter_K, :, :] = 0
                raw_compatibilities = raw_compatibilities.view(-1, H * W, self.filter_size)                             # [B,HW,K^2]
                compatabilities = self.masked_softmax(
                    self.softmax_temp * raw_compatibilities,
                    global_mask[i, j].unsqueeze(0).unsqueeze(-1),
                    dim=1,
                    epsilon=1e-5
                    )
                W_g = torch.einsum('bge,bgl->ble', X_g_vectors, compatabilities)                                        # [B,HW,C] x [B,HW,K^2] -> [B,K^2,C]
                X_l_flat_spatial = X_l.reshape(-1, self.filter_size, self.spatial_C)                                    # [B,K^2,C]
                forget_gate = torch.sigmoid(torch.sum(W_g * X_l_flat_spatial, dim=-1)).unsqueeze(-1)                    # [B,K^2,1]
                output[:, i, j] = (forget_gate * X_l_flat_spatial).sum(dim=1)                                           # [B,K^2,1] x [B,K^2,C] -> [B,K^2,C] -> [B,C]
        return output

    def forward_on_approach4(self, batch):
        global_mask = (1 - self.padding_mask - self.local_mask).flatten(start_dim=1, end_dim=-1)                        # [Nc,1,H,W,1] -> [Nc,HW,1]
        batch_pos = self.maybe_add_positional_encodings(batch)
        X_l, X_g = self.split_input(batch_pos, mask_X_g=False)

        keys = self.key_transform(X_g)                                                                                  # [1,B,HW,E]
        queries = self.query_transform(X_l)                                                                             # [F,B,K^2,E]
        values = self.value_transform(X_g)                                                                              # [1,B,HW,1]

        raw_compatibilities = self.cosine_similarity(
            keys, queries                                                                                               # [1,B,HW,E], [F,B,K^2,E]
        )                                                                                                               # [F,B,HW,K^2]
        compatabilities = self.masked_softmax(
            self.softmax_temp * raw_compatibilities,
            global_mask.unsqueeze(1).unsqueeze(-1),
            dim=2
        )
        elem_mul = values * compatabilities                                                                             # [1,B,HW,1] x [F,B,HW,K^2] -> [F,B,HW,K^2]
        W_g = elem_mul.sum(dim=2).unsqueeze(-1)                                                                         # [F,B,HW,K^2] -> [F,B,K^2] -> [F,B,K^2,1]
        X_l = X_l[:, :, :, :self.spatial_C]                                                                             # [F,B,K^2,E] -> [F,B,K^2,C]
        convolved_X = (W_g * X_l).sum(dim=2).permute(1,0,2)                                                             # [F,B,K^2,1] x [F,B,K^2,C] -> [F,B,K^2,C] -> [F,B,C] -> [B,F,C]
        output = convolved_X.reshape(
            -1, self.convs_height, self.convs_width, self.spatial_C
        )                                                                                                               # [B,F,C] -> [B,F_H,F_W,C]
        return output

    def forward_on_approach4_mem_efficient(self, batch):
        global_mask = (1 - self.padding_mask - self.local_mask).flatten(                                                # [Nc,1,H,W,1]
            start_dim=1, end_dim=-1).reshape(                                                                           # [Nc, HW]
                self.convs_height, self.convs_width, -1)                                                                # [F,F,HW]
        X = self.maybe_add_positional_encodings(batch)                                                                  # [B,H,W,E]
        batch_size, H, W, _ = X.shape
        keys = torch.nn.functional.normalize(self.key_transform(X), dim=-1)
        all_queries = torch.nn.functional.normalize(self.query_transform(X), dim=-1)
        X_flat_spatial = X.view(-1, H * W, X.shape[-1])                                                                 # [B,HW,E]
        values = self.value_transform(X_flat_spatial)                                                                   # [B,HW,1]

        convs_height = (H - self.filter_K) // self.stride + 1
        convs_width = (W - self.filter_K) // self.stride + 1

        output = torch.zeros(batch_size, convs_height, convs_width, self.spatial_C, dtype=torch.float).cuda()           # [B,F,F,C]
        for i in range(0, convs_height, self.stride):
            for j in range(0, convs_width, self.stride):
                queries = all_queries[:, i:i+self.filter_K, j:j+self.filter_K]                                          # [B,K,K,E]
                raw_compatibilities = torch.einsum('bhwe,bkje->bhwkj', keys, queries)                                   # [B,H,W,K,K]
                raw_compatibilities[:, i:i+self.filter_K, j:j+self.filter_K, :, :] = 0
                raw_compatibilities = raw_compatibilities.view(-1, H * W, self.filter_size)                             # [B,HW,K^2]
                compatabilities = self.masked_softmax(
                    self.softmax_temp * raw_compatibilities,
                    global_mask[i, j].unsqueeze(0).unsqueeze(-1),
                    dim=1,
                    epsilon=1e-5
                    )
                elem_mul = values * compatabilities                                                                     # [B,HW,1] x [B,HW,K^2] -> [B,HW,K^2]
                W_g = elem_mul.sum(dim=1).view(-1, self.filter_K, self.filter_K).unsqueeze(-1)                          # [B,HW,K^2] -> [B,K^2] -> [B,K,K] -> [B,K,K,1]
                X_l = X[:, i:i+self.filter_K, j:j+self.filter_K, :self.spatial_C]                                       # [B,K,K,C]
                output[:, i, j] = (W_g * X_l).sum(dim=(1,2))                                                            # [B,K,K,1] x [B,K,K,C] -> [B,K,K,C] -> [B,C]
        return output

    def forward_on_approach5(self, batch):
        batch = self.maybe_add_positional_encodings(batch)
        global_mask = (1 - self.padding_mask.reshape(1, -1)) - self.local_mask                                          # [Nc,HW]
        queries = self.query_transform(batch).flatten(1, 2)                                                             # [B,HW,C]
        values = self.value_transform(batch).flatten(1, 2)                                                              # [B,HW,C]
        keys = self.key_transform(batch).flatten(1, 2)                                                                  # [B,HW,C]

        cos_sim = self.cosine_similarity(queries, values)                                                               # [B,HW,HW]
        global_sims = torch.einsum('ijk,kl->ijl', cos_sim, global_mask.T)                                               # [B,HW,Nc]
        nand_indicator = F.sigmoid(-global_sims)                                                                        # [B,HW,Nc]
        local_nands = nand_indicator * self.local_mask.transpose(1, 0).unsqueeze(0)                                     # [B,HW,Nc]
        grouped_localities = torch.bmm(keys.permute(0, 2, 1), local_nands)                                              # [B,C,HW]x[B,HW,F]
        output = grouped_localities.permute(0, 2, 1).reshape(-1, self.convs_height, self.convs_width, self.spatial_C)   # [B,C,F] -> [B,F,C] -> [B,F,F,]
        return output

    # adapted from https://github.com/sooftware/attentions/blob/master/attentions.py
    def forward_on_self_attention(self, batch):
        batch_pos = self.maybe_add_positional_encodings(batch)
        queries = self.query_transform(batch_pos).flatten(1, 2)                                                         # [B,HW,C]
        values = self.value_transform(batch_pos).flatten(1, 2)                                                          # [B,HW,C]
        keys = self.key_transform(batch_pos).flatten(1, 2)                                                              # [B,HW,C]

        score = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(queries.shape[-1])                                 # [B,HW,C] x [B,C,HW] -> [B,HW,HW]
        attn = F.softmax(score, -1)                                                                                     # [B,HW,HW]
        context = torch.bmm(attn, values)                                                                               # [B,HW,HW] x [B,HW,C] -> [B,HW,C]
        return context.reshape(-1, self.input_H, self.input_W, self.spatial_C)                                          # [B,HW,C] -> [B,H,W,C]

    def forward_on_reverse_self_attention(self, batch):
        batch_pos = self.maybe_add_positional_encodings(batch)
        queries = self.query_transform(batch_pos).flatten(1, 2)                                                         # [B,HW,C]
        values = self.value_transform(batch_pos).flatten(1, 2)                                                          # [B,HW,C]
        keys = self.key_transform(batch_pos).flatten(1, 2)                                                              # [B,HW,C]
        # pdb.set_trace()
        score = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(queries.shape[-1])                                 # [B,HW,C] x [B,C,HW] -> [B,HW,HW]
        if self.approach_name == 'reversed_self_attention_normal':
            joint_scores = score.mean(-1, keepdim=True)                                                                 # [B,HW,1]
            expectation_fn = F.softmax(joint_scores, 1)                                                                 # [B,HW,1]
        elif self.approach_name == 'reversed_self_attention_elemwise':
            attn = F.softmax(score, 1)                                                                                  # [B,HW,HW]
            expectation_fn = torch.bmm(attn, values)                                                                    # [B,HW,HW] x [B,HW,C] -> [B,HW,C]
        
        context = expectation_fn * values                                                                               # [B,HW,C] x [B,HW,C] -> [B,HW,C]
        return context.reshape(-1, self.input_H, self.input_W, self.spatial_C)                                          # [B,H,W,C]

    def forward_on_background_self_attention(self, batch):
        # pdb.set_trace()
        batch_pos = self.maybe_add_positional_encodings(batch)

        global_mask = (1 - self.padding_mask.reshape(1, -1)) - self.local_mask                                          # [Nc,HW]
        scaled_batch = (batch_pos * self.elem_scales).flatten(1, 2)                                                     # [B,H,W,C] x [B,H,W,C] -> [B,HW,C]
        scaled_global = torch.matmul(scaled_batch.permute(0, 2, 1), global_mask.transpose(1,0)).permute(0, 2, 1)        # ([B,HW,C] -> [B,C,HW]) x ([Nc,HW] -> [HW,Nc]) -> [B,C,Nc] -> [B,Nc,C]
        
        global_queries = self.query_transform(scaled_global)                                                            # [B,Nc,C]
        local_values = self.value_transform(batch_pos).flatten(1, 2)                                                    # [B,HW,C]
        local_keys = self.key_transform(batch_pos).flatten(1, 2)                                                        # [B,HW,C]

        global_queries = F.normalize(global_queries, dim=-1)
        local_keys = F.normalize(local_keys, dim=-1)
        score = torch.bmm(global_queries, local_keys.transpose(1, 2)) / math.sqrt(global_queries.shape[-1])             # [B,Nc,C] x ([B,HW,C] -> [B,C,HW]) -> [B,Nc,HW]
        attn = self.masked_softmax(vec=score, mask=self.local_mask.unsqueeze(0), dim=-1)                                # [B,Nc,HW]
        convolution = torch.bmm(attn, local_values)                                                                     # [B,Nc,HW] x [B,HW,C] -> [B,Nc,C]
        if convolution.isnan().sum() > 0: 
            pdb.set_trace()
            print('Crap!')
        return convolution.reshape(-1, self.input_H, self.input_W, self.spatial_C)

    def forward(self, batch):
        """
        Input shape expected to be [B,C,H,W]
        """
        # pdb.set_trace()
        padded_batch = batch.permute(0, 3, 1, 2)
        padded_batch = self.input_padder(padded_batch)                                                                                # Pad batch for resolution reduction/preservation
        padded_batch = padded_batch.permute(0, 2, 3, 1)                                                                               # [B,C,H,W] -> [B,H,W,C]
        output = self.name2approach[self.approach_name](padded_batch)                                                          # [B,C,F,F] -> [B,F,F,C]
        # output = output.permute(0, 3, 1, 2)                                                                             # [B,F,F,C] -> [B,C,F,F]
        if self.use_residual_connection:
            residual = batch
            output = self.upsampler(output) + residual
        return output


class ConvAttnWrapper(nn.Module):
    def __init__(self, backbone, variant_kwargs):
        super().__init__()
        self.variant_kwargs = variant_kwargs
        # instantiate the backbone
        self.backbone = backbone
        # Obtain ordered list of backbone layers | Layer spatial information
        self.backbone_layers, self.backbone_spatial_shapes = self.backbone.get_network_structure()

        self.architecture = self.inject_variant()

    def inject_variant(self):
        injection_instructions = self.variant_kwargs['injection_info']
        self.network_structure = []
        start_point = 0
        for injection_instruction in injection_instructions:
            inject_layer, inject_number, filter_size = injection_instruction
            # Add backbone layers up to injection point
            self.network_structure += self.backbone_layers[start_point: inject_layer]
            spatial_shape = self.backbone_spatial_shapes[inject_layer]
            # Stack network modules
            for i in range(inject_number):
                variant_module = ConvolutionalSelfAttention(
                    spatial_shape=spatial_shape,
                    filter_size=filter_size,
                    approach_args=self.variant_kwargs
                )
                variant_bn = nn.BatchNorm2d(spatial_shape[-1])
                self.network_structure += [variant_module, variant_bn]
                spatial_shape = variant_module.get_output_shape()

            start_point = inject_layer
        self.network_structure += self.backbone_layers[start_point:]
        network_structure = nn.Sequential(*self.network_structure)
        return network_structure

    def forward(self, batch):
        # print('{}'.format(batch.permute(0, 3, 2, 1).shape[1:]))
        # for idx, layer in enumerate(self.network_structure):
        #     batch = layer(batch)
        #     print('{}'.format(batch.permute(0, 3, 2, 1).shape[1:]))
        # exit()
        # return batch
        return self.architecture(batch)
