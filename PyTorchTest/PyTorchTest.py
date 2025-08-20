
import torch
# import torch.nn as nn

# import faulthandler
# import signal
# faulthandler.register(signal.SIGUSR1.value)




if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Matrix multiplication example
print("Matmul example")
M = 4096
K = 4096
N = 4096

A = torch.ones((M, K)).to(device)
B = torch.ones((K, N)).to(device)
#C = torch.matmul(A,B)
#print(C.cpu())




# # Example model
# class Generate(nn.Module):
#     def __init__(self):
#         super(Generate, self).__init__()
#         self.gen = nn.Sequential(
#             nn.Linear(5,1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.gen(x)

# model = Generate() # Initialize the model
# model.to('cuda') # Move the model to the GPU

# # Create input data inside GPU
# input_data = torch.randn(16, 5, device=device)
# output = model(input_data) # Forward pass on theGP
# output