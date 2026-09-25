"""Linear heads and optimizer implementations shared by both FPC experiments."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BatchedLinearHeads(nn.Module):
    def __init__(self, n_models, feature_dim, num_classes):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(n_models, num_classes, feature_dim, device=DEVICE)
        )
        self.bias = nn.Parameter(
            torch.zeros(n_models, num_classes, device=DEVICE)
        )

    def forward(self, x):
        # x: [K, B, D] -> logits: [K, B, C]
        return torch.bmm(x, self.weight.transpose(1, 2)) + self.bias[:, None, :]

def fine_tune_batched(
    model,
    x,
    y,
    lr,
    epochs,
    batch_size,
    optimizer_name,
    weight_decay,
    seed,
    lbfgs_lr=1.0,
    lbfgs_max_iter=100,
    lbfgs_history_size=100,
    lbfgs_tolerance_grad=1e-7,
    lbfgs_tolerance_change=1e-9,
):
    # Independent LBFGS histories/line searches prevent coupling between heads.
    if optimizer_name == "LBFGS":
        for k in range(x.shape[0]):
            head = nn.Linear(x.shape[2], model.weight.shape[1]).to(DEVICE)
            with torch.no_grad():
                head.weight.copy_(model.weight[k])
                head.bias.copy_(model.bias[k])
            optimizer = torch.optim.LBFGS(
                head.parameters(), lr=lbfgs_lr, max_iter=lbfgs_max_iter,
                history_size=lbfgs_history_size, tolerance_grad=lbfgs_tolerance_grad,
                tolerance_change=lbfgs_tolerance_change, line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(head(x[k]), y[k])
                if weight_decay > 0:
                    loss = loss + 0.5 * weight_decay * head.weight.pow(2).sum()
                loss.backward()
                return loss

            optimizer.step(closure)
            with torch.no_grad():
                model.weight[k].copy_(head.weight)
                model.bias[k].copy_(head.bias)
            del optimizer, head
        return model

    # x: [K, N, D], y: [K, N]
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[150, 300, 400],
            gamma=0.2,
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    index_loader = DataLoader(
        torch.arange(x.shape[1]),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    model.train()
    for _ in range(epochs):
        for idx in index_loader:
            idx = idx.to(DEVICE)
            xb = x[:, idx, :]
            yb = y[:, idx]

            optimizer.zero_grad()
            logits = model(xb)
            K, B, C = logits.shape

            losses = F.cross_entropy(
                logits.reshape(K * B, C),
                yb.reshape(K * B),
                reduction="none",
            ).view(K, B)

            # Sum independent per-model objectives so each head gets the
            # same gradient it would get if trained separately.
            loss = losses.mean(dim=1).sum()
            if weight_decay > 0:
                loss = loss + 0.5 * weight_decay * model.weight.pow(2).sum()

            loss.backward()
            optimizer.step()
        if optimizer_name == "Adam":
            scheduler.step()

    return model


def add_lbfgs_arguments(parser):
    parser.add_argument("--lbfgs_lr", type=float, default=1.0)
    parser.add_argument("--lbfgs_max_iter", type=int, default=100)
    parser.add_argument("--lbfgs_history_size", type=int, default=100)
    parser.add_argument("--lbfgs_tolerance_grad", type=float, default=1e-07)
    parser.add_argument("--lbfgs_tolerance_change", type=float, default=1e-09)


def lbfgs_options(args):
    return {name: value for name, value in vars(args).items() if name.startswith("lbfgs_")}
