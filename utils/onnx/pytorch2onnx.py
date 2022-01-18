import torch
import numpy as np


def to_onnx(model, input_shape, dynamic_shape=True, onnx_path="model.onnx"):
    model.eval()
    dummy_input = torch.ones(*input_shape, dtype=torch.float32)

    input_names=['input']
    output_names=['output']
    if dynamic_shape:
        dynamic_axes= {'input': {0:'batch_size'}, 'output': {0:'batch_size'}} # define batch_size as dynamic input
    else:
        dynamic_axes = None
    
    torch.onnx.export(model, dummy_input, onnx_path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)


if __name__ == "__main__":
    # Test example

    # ---------------------------------------
    # Pytorch model (use yours insted)
    # ---------------------------------------
    class Net(torch.nn.Module):
        def __init__(self, num_clasess=1):
            super(Net, self).__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3, 2)
            self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
            self.fc = torch.nn.Linear(8, num_clasess)
        def forward(self, x):        
            out = self.pool(self.conv(x))
            out = out.view(-1, 8)
            out = self.fc(out)
            return out

    model = Net(num_clasess=3)
    # ---------------------------------------
    # Specify max_batch_size and input shape
    # NOTA: In case of using dynamic batch shape
    # it doesnt matters the batch_size, could be 1
    batch_size = 1
    color_channels = 3
    width = 32
    height = 32
    input_shape = (batch_size, color_channels, width, height)
    to_onnx(model, input_shape)
    