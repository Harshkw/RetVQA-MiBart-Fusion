# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CrossAttention(nn.Module):
#     def __init__(self, query_dim, key_value_dim):
#         super(CrossAttention, self).__init__()
#         self.query_projection = nn.Linear(query_dim, key_value_dim)  # Projection for query
#         self.key_projection = nn.Linear(key_value_dim, key_value_dim)  # Projection for keys
#         self.value_projection = nn.Linear(key_value_dim, key_value_dim)  # Projection for values
#         self.softmax = nn.Softmax(dim=-1)
#         self.scale_factor = torch.sqrt(torch.tensor(float(key_value_dim)))  # Scaling factor for attention scores

#     def forward(self, query, keys):
#         # Project query, keys, and values to the same dimensional space
#         Q = self.query_projection(query)  # Shape: [1, key_value_dim]
#         K = self.key_projection(keys)      # Shape: [num_images, key_value_dim]
#         V = self.value_projection(keys)    # Shape: [num_images, key_value_dim]

#         # Compute dot-product attention scores
#         attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale_factor  # Shape: [1, num_images]

#         # Normalize attention scores using softmax
#         attn_weights = self.softmax(attn_scores)  # Shape: [1, num_images]

#         # Apply attention weights to the values (weighted sum of values)
#         output = torch.matmul(attn_weights, V)  # Shape: [1, key_value_dim]

#         return output.squeeze(0), attn_weights














import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, num_heads=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = key_value_dim // num_heads
        assert key_value_dim % num_heads == 0, "key_value_dim must be divisible by num_heads"

        # Multi-head linear projections
        self.query_projection = nn.Linear(query_dim, key_value_dim)
        self.key_projection = nn.Linear(key_value_dim, key_value_dim)
        self.value_projection = nn.Linear(key_value_dim, key_value_dim)

        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(key_value_dim, key_value_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(key_value_dim * 2, key_value_dim)
        )

        self.output_projection = nn.Linear(key_value_dim, key_value_dim)
        self.norm1 = nn.LayerNorm(key_value_dim)
        self.norm2 = nn.LayerNorm(key_value_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale_factor = math.sqrt(self.head_dim)

    def forward(self, query, keys):
        batch_size = query.size(0)  # Usually 1

        # Projections
        Q = self.query_projection(query)  # [1, key_value_dim]
        K = self.key_projection(keys)     # [num_images, key_value_dim]
        V = self.value_projection(keys)   # [num_images, key_value_dim]

        # Reshape for multi-head: [batch, heads, seq_len, head_dim]
        Q = Q.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)        # [heads, 1, head_dim]
        K = K.view(-1, self.num_heads, self.head_dim).transpose(0, 1)                # [heads, num_images, head_dim]
        V = V.view(-1, self.num_heads, self.head_dim).transpose(0, 1)                # [heads, num_images, head_dim]

        # Attention scores: [heads, 1, num_images]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum: [heads, 1, head_dim]
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads: [1, key_value_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(batch_size, -1)

        # Add & Norm
        x = self.norm1(query + self.dropout(self.output_projection(attn_output)))  # residual connection

        # Feedforward with residual
        output = self.norm2(x + self.dropout(self.feedforward(x)))

        return output, attn_weights.mean(dim=0)  # Average attention weights across heads








