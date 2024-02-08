from torch.nn import Module
from torch_geometric.nn import GAT, MLP, Linear


class MEPOML_GAT(Module):
    def __init__(
        self,
        in_size=226,
        hid_size=700,
        num_conv=7,
        num_head=20,
        aggr_type="sum",
    ):
        super().__init__()

        self.in_mlp = MLP(
            [in_size, (in_size + hid_size) // 2, hid_size],
            plain_last=False,
        )

        self.gat = GAT(
            hid_size,
            hid_size,
            num_conv,
            v2=True,
            norm="batch_norm",
            heads=num_head,
            aggr=aggr_type,
        )

        out_size = 32
        self.out_mlp = MLP([hid_size, (hid_size + out_size) // 2, out_size])

        self.out_linear = Linear(out_size, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.in_mlp(x)
        x = self.gat(x, edge_index)
        x = self.out_mlp(x)
        y_raw = self.out_linear(x)
        y_err = y_raw.sum() / len(y_raw)
        y_fin = y_raw - y_err
        return y_fin
