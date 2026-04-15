import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class Conv1dPadSame(nn.Module):
    """扩展 nn.Conv1d 以支持 SAME 填充"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups)

    def forward(self, x):
        net = x
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.conv(net)
        return net


class MaxPool1dPadSame(nn.Module):
    """扩展 nn.MaxPool1d 以支持 SAME 填充"""

    def __init__(self, kernel_size):
        super(MaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.max_pool(net)
        return net


class BasicBlock(nn.Module):
    """ResNet 基本块"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups, downsample,
                 use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.35)
        self.conv1 = Conv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=self.groups)

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.35)
        self.conv2 = Conv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=self.groups)

        self.max_pool = MaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        identity = x
        out = x

        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.max_pool(identity)

        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        out += identity
        return out


class MultiHeadAttention(nn.Module):
    """多头注意力机制，用于跨模态融合"""

    def __init__(self, embed_dim, num_heads, modal_dim):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.modal_dim = modal_dim

        # 查询来自主模态（1D 数据）
        self.query = nn.Linear(embed_dim, embed_dim)
        # 键和值来自新模态（7 维特征）
        self.key = nn.Linear(modal_dim, embed_dim)
        self.value = nn.Linear(modal_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim ** -0.5

class MultiModalAttention(nn.Module):
    def __init__(self, hidden_dim, extra_modal_dim):
        super(MultiModalAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(extra_modal_dim, hidden_dim)
        self.value = nn.Linear(extra_modal_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, rnn_out, extra_modal):
        # rnn_out: (batch_size, seq_len, hidden_dim)
        # extra_modal: (batch_size, extra_modal_dim)

        # 扩展extra_modal维度 -> (batch_size, seq_len, extra_modal_dim)
        extra_modal = extra_modal.unsqueeze(1)

        # Q: 由RNN输出得到
        Q = self.query(rnn_out)  # (batch_size, seq_len, hidden_dim)

        # K, V: 由额外模态得到
        K = self.key(extra_modal)  # (batch_size, seq_len, hidden_dim)
        V = self.value(extra_modal)  # (batch_size, seq_len, hidden_dim)

        # 注意力分数计算
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5)  # (batch_size, seq_len, seq_len)
        attn_weights = self.softmax(attn_scores)

        # 加权求和得到融合后的特征
        attn_output = torch.bmm(attn_weights, V)  # (batch_size, seq_len, hidden_dim)

        # 融合RNN输出和注意力结果
        fused_output = rnn_out + attn_output  # 残差连接
        return fused_output


class MultiHeadSelfAttention(nn.Module):
    """1D 数据的多头自注意力"""

    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        # 输入形状: (batch_size, channels, seq_len)
        batch_size, channels, seq_len = x.shape

        # 转置为 (batch_size, seq_len, channels) 用于注意力计算
        x = x.permute(0, 2, 1)

        # 计算 Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, embed_dim)
        K = self.key(x)  # (batch_size, seq_len, embed_dim)
        V = self.value(x)  # (batch_size, seq_len, embed_dim)

        # 重塑为多头注意力
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        # 重塑并投影
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.embed_dim)
        out = self.out(context)

        # 转置回 (batch_size, channels, seq_len)
        out = out.permute(0, 2, 1)

        return out

class RCNet(nn.Module):
    """带跨模态注意力和 LayerNorm 的 Res-Confidence 网络，适用于 1D 数据和 7 维新模态"""

    def __init__(self,
                 in_channels=1,
                 base_filters=10,
                 kernel_size=15,
                 stride=2,
                 groups=1,
                 n_block=8,
                 n_classes=10,
                 downsample_gap=2,
                 increasefilter_gap=4,
                 use_bn=True,
                 use_do=True,
                 num_heads=4,
                 modal_dim=7):
        super().__init__()

        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.downsample_gap = downsample_gap
        self.increasefilter_gap = increasefilter_gap
        self.num_heads = num_heads
        self.modal_dim = modal_dim

        # 初始卷积块
        self.first_block_conv = Conv1dPadSame(
            in_channels=in_channels,
            out_channels=base_filters,
            kernel_size=self.kernel_size,
            stride=1
        )
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()

        # LayerNorm 和跨模态注意力
        self.pre_attn_norm = nn.LayerNorm(base_filters)
        self.self_attention = MultiHeadSelfAttention(embed_dim=base_filters, num_heads=self.num_heads)
        self.attention = MultiModalAttention(1500,7)
        self.attention1 = MultiModalAttention(1500,14)
        self.attention2 = MultiModalAttention(1500,32)
        self.post_attn_norm = nn.LayerNorm(base_filters)
        self.dropout = nn.Dropout(p=0)

        # 残差块
        self.basicblock_list = nn.ModuleList()
        out_channels = base_filters
        for i_block in range(self.n_block):
            is_first_block = (i_block == 0)
            downsample = (i_block % self.downsample_gap == 1)

            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                in_channels = int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=downsample,
                use_bn=self.use_bn,
                use_do=self.use_do,
                is_first_block=is_first_block
            )
            self.basicblock_list.append(tmp_block)

        # 最终预测层
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense_ = nn.Linear(out_channels, n_classes)
        self.dense2_ = nn.Linear(out_channels, 2)
        self.dense_abc_angles = nn.Linear(out_channels, 6)
        self.softmax = nn.Softmax(dim=1)
        # self.linear = nn.Linear(1500,1500)
        # self.linear1 = nn.Linear(1500,1500)
        # self.linear2 = nn.Linear(1500,1500)
        # self.linear3 = nn.Linear(1500,1500)
        # self.linear4 = nn.Linear(1500,1500)

    def forward(self, x, modal, modal1,modal2):
        out = x
        # modal = torch.cat([modal, modal1,modal2], dim=1)
        # 初始卷积
        out = self.first_block_conv(out)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        # # 跨模态注意力，带 LayerNorm 和残差连接
        # residual = out
        # out1 = self.pre_attn_norm(out.permute(0, 2, 1)).permute(0, 2, 1)  # 归一化通道维度
        # out1 = self.attention(out1, modal)
        # out1 = self.post_attn_norm(out1.permute(0, 2, 1)).permute(0, 2, 1)  # 归一化通道维度
        # #
        # residual = out
        # out2 = self.pre_attn_norm(out.permute(0, 2, 1)).permute(0, 2, 1)  # 归一化通道维度
        # out2 = self.attention1(out2, modal1)
        # out2 = self.self_attention(out2)
        # out2 = self.post_attn_norm(out2.permute(0, 2, 1)).permute(0, 2, 1)  # 归一化通道维度
        #
        # residual = out
        # out3 = self.pre_attn_norm(out.permute(0, 2, 1)).permute(0, 2, 1)  # 归一化通道维度
        # out3 = self.attention2(out3, modal2)
        # out3 = self.self_attention(out3)
        # out3 = self.post_attn_norm(out3.permute(0, 2, 1)).permute(0, 2, 1)  # 归一化通道维度
        #
        #
        # out = out1+out2+out3+residual



        # 残差块
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            out = net(out)

        # 最终预测
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)  # 全局平均池化
        c = self.softmax(self.dense2_(out))  # 置信度分数
        abc_angles = self.dense_abc_angles(out)
        out = self. dense_(out)  # 分类 logits
        # out = self.softmax(out)  # 分类概率


        return out,c,abc_angles


# 测试模型
def test_rcnet():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)

    # 创建主模态输入张量，形状 (256, 1500)
    x = torch.randn(256, 1500)
    x = x.unsqueeze(1)  # 重塑为 (256, 1, 1500)

    # 创建新模态输入，7 维特征，形状 (256, 7)
    modal = torch.randn(256, 7)

    print(f"主模态输入形状: {x.shape}")
    print(f"新模态输入形状: {modal.shape}")

    # 实例化模型
    model = RCNet(
        in_channels=1,
        base_filters=10,
        kernel_size=15,
        stride=2,
        groups=1,
        n_block=8,
        n_classes=10,
        downsample_gap=2,
        increasefilter_gap=4,
        use_bn=True,
        use_do=True,
        num_heads=2,
        modal_dim=7
    )

    # 设置模型为评估模式
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dvsid = [0, 1, 2, 3]

    model = nn.DataParallel(model, output_device=dvsid).to(device)  #model #
    x = x.to(device)
    modal = modal.to(device)

    # 前向传播
    with torch.no_grad():  # 禁用梯度计算以进行测试
        out= model(x, modal)
    print(f"输出形状（分类概率）: {out.shape}")
    # print(f"输出形状（置信度分数）: {c.shape}")
    print(f"分类概率和为 1: {torch.allclose(out.sum(dim=1), torch.ones(256))}")
    # print(f"置信度分数在 [0, 1]: {(c >= 0).all() and (c <= 1).all()}")
    print("模型测试成功！")


if __name__ == "__main__":
    test_rcnet()