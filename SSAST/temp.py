import torch

checkpoint_path = "C:\\Users\\ander\\Github\\Masters\\Results_from_cluster\\pretrained_best.pth.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda"))

# Extract only the model weights
model_weights = checkpoint["state_dict"]
#print(model_weights)

# Remove PyTorch Lightning prefixes (e.g., "model.")
cleaned_weights = {k.replace("model.", ""): v for k, v in model_weights.items()}

# Save cleaned weights as a standard PyTorch checkpoint
torch.save(cleaned_weights, "../Results_from_cluster/ssast/ssast_model.pth")