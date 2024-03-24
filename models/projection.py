import torch.nn as nn

# Define the architecture of your MLP projection layer
class MLPProjection(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MLPProjection, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Define list to hold linear layers
        self.linear_layers = nn.ModuleList()
        # Add the first linear layer
        self.linear_layers.append(nn.Linear(input_size, hidden_size))
        # Add additional linear layers
        for _ in range(num_layers - 1):
            self.linear_layers.append(nn.Linear(hidden_size, hidden_size))
        # GELU activation function
        self.activation = nn.GELU()

    def forward(self, x):
        for linear_layer in self.linear_layers:
            x = self.activation(linear_layer(x))
        return x