from train_data.wikidataset import WikiDataLoader
from models.hyena.hyena import HyenaModel
import torch
from captum.attr import LayerConductance, IntegratedGradients, LayerIntegratedGradients
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", model_max_length=256, padding=True)
file = "train_data/wikipedia_10k.txt"
data_loader = WikiDataLoader(file, tokenizer,block_size=256)
dataloader = data_loader.get_dataloader()
input_ids = next(iter(dataloader))
model = HyenaModel(tokenizer.vocab_size,256, tokenizer.model_max_length)

def hyena_forward_func(input_ids, target_index=None):
    # Ensure input_ids are of type torch.long
    input_ids = input_ids.long()
    model_output = model(input_ids)

    # If a target_index is specified, select the output at that index
    if target_index is not None:
        target_output = model_output[:, target_index]
    else:
        raise ValueError("Target index not provided for attribution")

    return target_output

# Initialize LayerIntegratedGradients with the hyena_forward_func and the embedding layer
# Assuming model is your HyenaModel and input_ids is a batch of input tokens
lig = LayerIntegratedGradients(hyena_forward_func, model.embedding)

# The baseline should be the same shape as input_ids and consist of the padding token index
ref_input_ids = torch.full(input_ids.shape, tokenizer.pad_token_id, dtype=torch.long)

# Initialize a tensor to store the attributions for each token
all_attributions = torch.zeros(input_ids.size(0), input_ids.size(1), dtype=torch.float)

# Compute attributions for each token in the sequence
all_attributions = torch.zeros(input_ids.size(0), input_ids.size(1), dtype=torch.float)

# Compute attributions for each token in the sequence
for target_index in range(input_ids.size(1)):
    attributions = lig.attribute(
        input_ids,
        baselines=ref_input_ids,
        additional_forward_args=(target_index,),
        target=target_index
    )

    # Sum over the embedding dimension to get one score per token
    # The resulting shape should be (batch_size, num_tokens)
    summed_attributions = attributions.sum(dim=2)

    # No need to squeeze as the shape is already (batch_size, num_tokens)
    if summed_attributions.ndim == 2 and summed_attributions.shape[1] == input_ids.size(1):
        # Assign the summed attributions to the corresponding index in all_attributions
        all_attributions[:, target_index] = summed_attributions[:, target_index]
    else:
        raise RuntimeError(f"Unexpected shape for summed_attributions: {summed_attributions.shape}")

# Move the attributions to CPU for visualization
all_attributions = all_attributions.cpu().detach().numpy()

import numpy as np
import matplotlib.pyplot as plt

def visualize_token_attributions(attributions, token_list):
    # Ensure attributions is a 2D tensor by summing across the embedding dimension
    if attributions.ndim == 3:
        attributions = attributions.sum(dim=-1)

    # Convert attributions to numpy for visualization
    attributions_np = attributions.cpu().detach().numpy()

    # Determine the size of the largest dimension
    max_length = max(attributions_np.shape[1], len(token_list))

    # Initialize a square matrix filled with zeros
    square_matrix = np.zeros((max_length, max_length))

    # Fill the part of the square matrix with actual attribution scores
    square_matrix[:attributions_np.shape[1], :attributions_np.shape[1]] = attributions_np[0]

    fig, ax = plt.subplots(figsize=(20, 20))

    # Create a heatmap of token attributions
    im = ax.matshow(square_matrix, cmap='viridis')

    # Set ticks and labels
    tick_positions = np.linspace(0, len(token_list) - 1, num=50, dtype=int)  # Only 50 evenly spaced tick positions
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(np.array(token_list)[tick_positions], rotation=90, fontsize=8)
    ax.set_yticklabels(np.array(token_list)[tick_positions], fontsize=8)

    # Add colorbar to the heatmap
    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    plt.xlabel('Tokens')
    plt.ylabel('Tokens')
    plt.tight_layout()
    plt.show()
all_tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0].tolist()]
# Assuming attributions is the tensor you got from Layer Integrated Gradients
# Assuming all_tokens is the list of tokens corresponding to input_ids
visualize_token_attributions(attributions[0], all_tokens)

