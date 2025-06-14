import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.ep = 1e-3

    def forward(self, question_embeddings_pos, question_embeddings_neg, pos_image_embeddings, neg_image_embeddings, batch_size):
        # Compute similarities
        pos_similarities = self.cosine_similarity(question_embeddings_pos, pos_image_embeddings)
        neg_similarities = self.cosine_similarity(question_embeddings_neg, neg_image_embeddings)

        epos = torch.exp(pos_similarities)    #[batch_size*2]
        eneg = torch.exp(neg_similarities)    #[batch_size*9]
        total_loss = 0
        for i in range(batch_size):
            pos_samples = epos[2*i:2*i + 2]
            neg_samples = eneg[23*i:23*i + 23]
            #pos_loss = -torch.log(torch.exp(pos_samples) / (torch.exp(pos_samples).sum() + self.ep)).mean()
            #neg_loss = -torch.log(torch.exp(-neg_samples) / (torch.exp(-neg_samples).sum() + self.ep)).mean()
            Sum = torch.sum(pos_samples) + torch.sum(neg_samples) + self.ep
            total_loss += (torch.sum(neg_samples) - torch.sum(pos_samples))/Sum
            #loss = pos_loss + neg_loss
            #total_loss += loss

        return total_loss