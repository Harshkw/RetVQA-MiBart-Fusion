import torch
import torch.nn as nn

class AttentionLoss(nn.Module):
    def __init__(self):
        super(AttentionLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, attn_weights, num_positive, num_negative):
        # Attention weights for positive images
        pos_weights = attn_weights[0, :num_positive]
        # Attention weights for negative images
        neg_weights = attn_weights[0, num_positive:]

        # Create target labels: 1 for positive images, 0 for negative images
        # pos_targets = torch.ones_like(pos_weights)
        # neg_targets = torch.zeros_like(neg_weights)
        pos_targets = torch.ones_like(pos_weights) * 0.9  # Assign 0.9 to positive images
        neg_targets = torch.ones_like(neg_weights) * 0.1  # Assign 0.05 to negative images
        # Compute BCE loss for both positive and negative samples
        pos_loss = self.bce_loss(pos_weights, pos_targets)
        neg_loss = self.bce_loss(neg_weights, neg_targets)

        # Total loss is the sum of positive and negative BCE losses
        total_loss = pos_loss + neg_loss 

        return total_loss
    
class CustomLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(CustomLoss, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.ep = 1e-3

    def forward(self, question_embed, question_embed_next, pos_image_embeddings):
        # Compute similarities
        pos_similarities_1 = self.cosine_similarity(question_embed, pos_image_embeddings[0])

        pos_similarities_2 = self.cosine_similarity(question_embed, pos_image_embeddings[1])
        neg_similarities_1 = self.cosine_similarity(question_embed_next, pos_image_embeddings[0])
        neg_similarities_2 = self.cosine_similarity(question_embed_next, pos_image_embeddings[1])

        epos = torch.exp(pos_similarities_1) + torch.exp(pos_similarities_2)    #[batch_size*2]
        eneg = torch.exp(neg_similarities_1) + torch.exp(neg_similarities_2)   #[batch_size*9]
        total_loss = 0

        Sum = epos + eneg + self.ep
        total_loss = (eneg - epos)/Sum

        return total_loss[0]
