import torch
torch.manual_seed(0)  # Set the CPU random seed to 0
torch.cuda.manual_seed_all(0)  # Set the random seed for all GPUs to 0.
torch.backends.cudnn.deterministic = True  # Enable CuDNN deterministic algorithms (guaranteeing reproducible results)
torch.backends.cudnn.benchmark = False  # Disable CuDNN auto-optimization (to avoid introducing randomness)
import torch.nn as nn  # Import PyTorch neural network modules

#Define a Logistic Regression Model LogReg
class LogReg(nn.Module):
    """
    Logical classifier
    """

    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()  # Initialization method inheriting from parent class nn.Module
        self.fc = nn.Linear(ft_in, nb_classes)  # Define the fully connected layer

        # Iterate through all submodules and initialize weights.
        for m in self.modules():
            self.weights_init(m)

# Weight Initialization Method
    def weights_init(self, m):
        if isinstance(m, nn.Linear):  # Check whether the module is a linear layer
            torch.nn.init.xavier_uniform_(m.weight.data)  # Xavier uniform distribution weight initialization
            if m.bias is not None:
                m.bias.data.fill_(0.0)  # Bias initialized to zero

#Forward Propagation
    def forward(self, seq):
        ret = self.fc(seq)  # Input data passes through a fully connected layer.
        return ret  # Return output (without Softmax)