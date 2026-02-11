# nets/cph.py
import torch
from torch import nn
import torch.nn.functional as F
from thop import profile


def _pick_gn_groups(c: int, preferred=(32, 16, 8, 4, 2, 1)) -> int:
    for g in preferred:
        if c % g == 0:
            return g
    return 1


def make_norm(num_channels: int, kind: str = "gn", dims: str = "2d", gn_groups: int = 16):
    kind = (kind or "gn").lower()
    if kind == "gn":
        g = _pick_gn_groups(num_channels, (gn_groups, 32, 16, 8, 4, 2, 1))
        return nn.GroupNorm(g, num_channels)
    if kind == "syncbn":
        return nn.SyncBatchNorm(num_channels)
    if kind == "bn":
        return nn.BatchNorm2d(num_channels) if dims == "2d" else nn.BatchNorm1d(num_channels)
    if kind in ("none", "identity", "id"):
        return nn.Identity()
    raise ValueError(f"Unknown norm kind: {kind}")


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
        norm: str = "gn",
        gn_groups: int = 16,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=(norm in ["none", None]),
        )
        norm_layer = make_norm(out_channels, kind=norm, dims="2d", gn_groups=gn_groups)
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, norm_layer, relu)


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size,
        stride=1,
        padding=1,
        activation=True,
        norm: str = "gn",
        gn_groups: int = 16,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=(norm in ["none", None])
        )
        self.norm = make_norm(c_out, kind=norm, dims="2d", gn_groups=gn_groups)
        self.relu = nn.ReLU(inplace=True)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, cin, cout, norm: str = "gn", gn_groups: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1, norm=norm, gn_groups=gn_groups),
            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False, norm=norm, gn_groups=gn_groups),
        )
        self.conv1 = nn.Conv2d(cout, cout, 1, bias=(norm in ["none", None]))
        self.relu = nn.ReLU(inplace=True)
        self.norm = make_norm(cout, kind=norm, dims="2d", gn_groups=gn_groups)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.norm(x)
        x = h + x
        x = self.relu(x)
        return x


class DWCONV(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=None):
        super().__init__()
        if groups is None:
            groups = in_channels
        self.depthwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=True
        )

    def forward(self, x):
        return self.depthwise(x)


class UEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, norm: str = "gn", gn_groups: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.res1 = DoubleConv(in_channels, 64, norm=norm, gn_groups=gn_groups)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = DoubleConv(64, 128, norm=norm, gn_groups=gn_groups)
        self.pool2 = nn.MaxPool2d(2)
        self.res3 = DoubleConv(128, 256, norm=norm, gn_groups=gn_groups)
        self.pool3 = nn.MaxPool2d(2)
        self.res4 = DoubleConv(256, 512, norm=norm, gn_groups=gn_groups)
        self.pool4 = nn.MaxPool2d(2)
        self.res5 = DoubleConv(512, 1024, norm=norm, gn_groups=gn_groups)
        self.pool5 = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        x = self.res1(x); features.append(x); x = self.pool1(x)
        x = self.res2(x); features.append(x); x = self.pool2(x)
        x = self.res3(x); features.append(x); x = self.pool3(x)
        x = self.res4(x); features.append(x); x = self.pool4(x)
        x = self.res5(x); features.append(x); x = self.pool5(x)
        features.append(x)
        return features


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, norm: str = "gn", gn_groups: int = 16):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels, out_channels, kernel_size=3, padding=1, norm=norm, gn_groups=gn_groups
        )
        self.conv2 = Conv2dReLU(
            out_channels, out_channels, kernel_size=3, padding=1, norm=norm, gn_groups=gn_groups
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        up = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, up)


class HGNN(nn.Module):
    def __init__(self, in_ch, n_out, norm: str = "gn", gn_groups: int = 16):
        super().__init__()
        self.conv = nn.Linear(in_ch, n_out)
        self.norm = make_norm(n_out, kind=norm, dims="1d", gn_groups=gn_groups)

    def forward(self, x, G):
        residual = x
        x = self.conv(x)
        x = G.matmul(x)
        x = self.norm(x.permute(0, 2, 1).contiguous())
        x = F.relu(x).permute(0, 2, 1).contiguous() + residual
        return x


class G_HGNN_layer(nn.Module):
    def __init__(self, in_ch, node=None, K_neigs=None, kernel_size=5, stride=2, norm: str = "gn", gn_groups: int = 16):
        super().__init__()
        self.HGNN = HGNN(in_ch, in_ch, norm=norm, gn_groups=gn_groups)
        self.K_neigs = K_neigs
        self.node = node
        self.kernel_size = kernel_size
        self.stride = stride
        self.single_local_H = self.local_kernel(node, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        B, N, C = x.shape
        x_merged = x.reshape(B * N, C).unsqueeze(0)
        ori_dists = self.pairwise_distance(x_merged)
        k = self.K_neigs[0]
        topk_dists, topk_inds = ori_dists.topk(k + 1, dim=2, largest=False, sorted=True)
        avg_dists = ori_dists.mean(-1, keepdim=True)
        H = self.create_incidence_matrix_inter(topk_dists, topk_inds, avg_dists, B, N)
        Dv = torch.sum(H, dim=2, keepdim=True)
        alpha = 1.0
        Dv = Dv * alpha
        max_k = int(Dv.max())
        _topk_dists, _topk_inds = ori_dists.topk(max_k, dim=2, largest=False, sorted=True)
        _avg_dists = ori_dists.mean(-1, keepdim=True)
        new_H = self.create_incidence_matrix_inter(_topk_dists, _topk_inds, _avg_dists, B, N)
        local_H = self.build_block_diagonal_localH(self.single_local_H, B, x.device)
        _H = torch.cat([new_H, local_H], dim=2)
        _G = self._generate_G_from_H_b(_H)
        x_out = self.HGNN(x_merged, _G)
        x_out = x_out.squeeze(0).view(B, N, C)
        return x_out

    @torch.no_grad()
    def create_incidence_matrix_inter(self, top_dists, inds, avg_dists, B, N, prob=False):
        _, total_nodes, K = top_dists.shape
        weights = self.weights_function(top_dists, avg_dists, prob)
        incidence_matrix = torch.zeros(1, total_nodes, total_nodes, device=inds.device)
        pixel_indices = torch.arange(total_nodes, device=inds.device)[:, None]
        incidence_matrix[0, pixel_indices, inds.squeeze(0)] = weights.squeeze(0)
        return incidence_matrix

    def build_block_diagonal_localH(self, single_local_H, B, device):
        N = self.node * self.node
        E = single_local_H.size(1)
        H_local = single_local_H.to(device)
        block_diag = torch.zeros(B * N, B * E, device=device)
        for i in range(B):
            startN = i * N; endN = startN + N
            startE = i * E; endE = startE + E
            block_diag[startN:endN, startE:endE] = H_local
        return block_diag.unsqueeze(0)

    @torch.no_grad()
    def _generate_G_from_H_b(self, H, variable_weight=False):
        bs, n_node, n_hyperedge = H.shape
        W = torch.ones([bs, n_hyperedge], requires_grad=False, device=H.device)
        DV = torch.sum(H, dim=2)
        DE = torch.sum(H, dim=1).clamp_min_(1e-8)
        invDE = torch.diag_embed((torch.pow(DE, -1)))
        DV2 = torch.diag_embed((torch.pow(DV, -0.5)))
        W = torch.diag_embed(W)
        HT = H.transpose(1, 2)
        if variable_weight:
            DV2_H = DV2 @ H
            invDE_HT_DV2 = invDE @ HT @ DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2 @ H @ W @ invDE @ HT @ DV2
            return G

    @torch.no_grad()
    def pairwise_distance(self, x):
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)

    @torch.no_grad()
    def weights_function(self, topk_dists, avg_dists, prob):
        if prob:
            topk_dists_sq = topk_dists.pow(2)
            normalized_topk_dists_sq = topk_dists_sq / avg_dists
            weights = torch.exp(-normalized_topk_dists_sq)
        else:
            weights = torch.ones(topk_dists.size(), device=topk_dists.device)
        return weights

    @torch.no_grad()
    def local_kernel(self, size, kernel_size=3, stride=1):
        inp = torch.arange(size * size, dtype=torch.float).reshape(size, size)[None, None, :, :]
        inp_unf = F.unfold(inp, kernel_size=(kernel_size, kernel_size), stride=stride).squeeze(0).transpose(0, 1).long()
        edge, node = inp_unf.shape
        matrix = torch.arange(edge)[:, None].repeat(1, node).long()
        H_local = torch.zeros((size * size, edge))
        H_local[inp_unf, matrix] = 1.
        return H_local


class G_HyperNet(nn.Module):
    def __init__(self, channel, node=28, kernel_size=3, stride=1, K_neigs=None, norm: str = "gn", gn_groups: int = 16):
        super().__init__()
        self.G_HGNN_layer = G_HGNN_layer(channel, node=node, kernel_size=kernel_size, stride=stride,
                                         K_neigs=K_neigs, norm=norm, gn_groups=gn_groups)

    def forward(self, x):
        b, c, w, h = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1).contiguous()
        x = self.G_HGNN_layer(x)
        x = x.permute(0, 2, 1).contiguous().view(b, c, w, h)
        return x


class HyperEncoder(nn.Module):
    def __init__(self, channel=[1024, 1024], norm: str = "gn", gn_groups: int = 16):
        super().__init__()
        kernel_size = 3
        stride = 1
        self.HGNN_layer2 = G_HyperNet(channel[0], node=14, kernel_size=kernel_size, stride=stride,
                                      K_neigs=[1], norm=norm, gn_groups=gn_groups)
        self.HGNN_layer3 = G_HyperNet(channel[1], node=7, kernel_size=kernel_size, stride=stride,
                                      K_neigs=[1], norm=norm, gn_groups=gn_groups)

    def forward(self, x):
        _, _, _, _, feature2, feature3 = x
        out2 = self.HGNN_layer2(feature2)
        out3 = self.HGNN_layer3(feature3)
        return [out2, out3]


class ParallEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, norm: str = "gn", gn_groups: int = 16):
        super().__init__()
        self.Encoder1 = UEncoder(in_channels=in_channels, norm=norm, gn_groups=gn_groups)
        self.Encoder2 = HyperEncoder(channel=[1024, 1024], norm=norm, gn_groups=gn_groups)
        self.num_module = 2
        self.fusion_list = [1024, 1024]
        self.squeelayers = nn.ModuleList([
            nn.Conv2d(self.fusion_list[i] * 2, self.fusion_list[i], kernel_size=1, stride=1)
            for i in range(self.num_module)
        ])

    def forward(self, x, return_hyper: bool = False):
        skips = []
        features = self.Encoder1(x)
        feature_hyper = self.Encoder2(features)
        skips.extend(features[:4])
        for i in range(self.num_module):
            fused = self.squeelayers[i](torch.cat((feature_hyper[i], features[i + 4]), dim=1))
            skips.append(fused)
        if return_hyper:
            return skips, feature_hyper
        return skips


class CPH(nn.Module):
    def __init__(self, n_classes: int = 9, in_channels: int = 4, norm: str = "gn", gn_groups: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.norm = norm
        self.gn_groups = gn_groups
        self.p_encoder = ParallEncoder(in_channels=in_channels, norm=norm, gn_groups=gn_groups)
        self.encoder_channels = [1024, 512, 256, 128, 64]
        self.decoder1 = DecoderBlock(self.encoder_channels[0] + self.encoder_channels[0],
                                     self.encoder_channels[1], norm=norm, gn_groups=gn_groups)
        self.decoder2 = DecoderBlock(self.encoder_channels[1] + self.encoder_channels[1],
                                     self.encoder_channels[2], norm=norm, gn_groups=gn_groups)
        self.decoder3 = DecoderBlock(self.encoder_channels[2] + self.encoder_channels[2],
                                     self.encoder_channels[3], norm=norm, gn_groups=gn_groups)
        self.decoder4 = DecoderBlock(self.encoder_channels[3] + self.encoder_channels[3],
                                     self.encoder_channels[4], norm=norm, gn_groups=gn_groups)
        self.segmentation_head2 = SegmentationHead(in_channels=256, out_channels=n_classes, kernel_size=1)
        self.segmentation_head3 = SegmentationHead(in_channels=128, out_channels=n_classes, kernel_size=1)
        self.segmentation_head4 = SegmentationHead(in_channels=64, out_channels=n_classes, kernel_size=1)
        self.segmentation_head5 = SegmentationHead(in_channels=64, out_channels=n_classes, kernel_size=1)
        self.decoder_final = DecoderBlock(in_channels=64, out_channels=64, norm=norm, gn_groups=gn_groups)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_feats: bool = False, return_graph: bool = False):
        if x.size(1) == 1 and self.in_channels == 4:
            x = x.repeat(1, 4, 1, 1)
        if x.size(1) != self.in_channels:
            raise ValueError(f"CPH expects {self.in_channels} input channels, got {x.size(1)}")
        pe_out = self.p_encoder(x, return_hyper=True)
        if isinstance(pe_out, tuple):
            encoder_skips, feature_hyper = pe_out
        else:
            encoder_skips = pe_out
            feature_hyper = None
        x1_up = self.decoder1(encoder_skips[-1], encoder_skips[-2])
        x2_up = self.decoder2(x1_up, encoder_skips[-3])
        x3_up = self.decoder3(x2_up, encoder_skips[-4])
        x4_up = self.decoder4(x3_up, encoder_skips[-5])
        x_final = self.decoder_final(x4_up, None)
        logits = self.segmentation_head5(x_final)
        if return_feats or return_graph:
            aux = {"feat_map": x1_up}
            if feature_hyper is not None:
                aux["struct_maps"] = (feature_hyper[0], feature_hyper[1])
            return logits, aux
        return logits


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CPH(n_classes=3, in_channels=4, norm='gn', gn_groups=16).to(device)
    inout = torch.randn((16, 4, 224, 224), device=device)
    logits, aux = model(inout, return_feats=True)
    print(logits.shape, aux["feat_map"].shape)
    print('# generator parameters (M):', 1.0 * sum(p.numel() for p in model.parameters()) / 1e6)
    macs, params = profile(model, inputs=(inout,))
    print("FLOPs: {:.2f} GFLOPs".format(macs / 1e9))
    print("Parameters: {:.2f} M".format(params / 1e6))
