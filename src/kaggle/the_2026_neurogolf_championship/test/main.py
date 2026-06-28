import torch
import torch.nn as nn


class ARC007BBFB7(nn.Module):
    def forward(self, x):
        # x shape = (1, 1, 3, 3)

        tiled = x.repeat(1, 1, 3, 3)

        mask = (
            (x != 0)
            .repeat_interleave(3, dim=2)
            .repeat_interleave(3, dim=3)
        )

        return tiled * mask.float()


model = ARC007BBFB7()

dummy = torch.randint(
    0,
    10,
    (1, 1, 3, 3)
).float()

torch.onnx.export(
    model,
    dummy,
    "arc007bbfb7.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
)