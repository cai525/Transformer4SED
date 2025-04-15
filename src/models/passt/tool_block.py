import torch
import torch.nn as nn


class GAP(nn.Module):
    def __init__(self):
        super(GAP, self).__init__()
        pass

    def forward(self, input):
        #input shape, B,T,C
        return torch.mean(input, dim=1)  #return B,C


class th_learner_block(nn.Module):
    def __init__(self, input_logit_dim=10, input_embedding_dim=1536, output_dim=10, middle_dim=128):
        super(th_learner_block, self).__init__()
        self.logit_projector = nn.Sequential(nn.Linear(input_logit_dim, middle_dim), nn.ReLU())
        self.embedding_projector = nn.Sequential(nn.Linear(input_embedding_dim, middle_dim),
                                                 nn.ReLU())
        self.th_learner = nn.Sequential(nn.Linear(middle_dim, middle_dim), nn.ReLU(),
                                        nn.Linear(middle_dim, middle_dim), nn.ReLU(),
                                        nn.Linear(middle_dim, output_dim))
        self.th_sigmoid = nn.Sigmoid()

    def forward(self, logit, embedding):
        #input dimension (B,C) or (B,T,C)
        logit = logit.detach()
        embed = embedding.detach()
        x1_at = self.logit_projector(logit)
        x2_at = self.embedding_projector(embed)

        th_learner_input = x1_at + x2_at
        th_learner_embedding = self.th_learner(th_learner_input)
        th_learner_output = self.th_sigmoid(th_learner_embedding)

        #input shape, B,T,C
        return th_learner_output