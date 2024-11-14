import torch

def getMostProbableToken(distribution: torch.Tensor):
    """
    Returns the token index with the highest probability and its corresponding probability.
    """
    token_index = distribution.argmax(dim=-1)
    token_prob = distribution.max(dim=-1).values
    return token_index, token_prob

def getTopKTokens(distribution: torch.Tensor, k: int, normalize: bool = False):
    """
    Returns the top k most probable tokens and their probabilities.
    """
    topKIndices = distribution.argsort(dim=-1)[..., -k:]
    topKValues = distribution.gather(dim=-1, index=topKIndices)

    if normalize:
        topKValues = topKValues / topKValues.sum(dim=-1, keepdim=True).to(torch.float32)

    return topKIndices, topKValues

def getATokenFromTopK(distribution: torch.Tensor, k: int):
    """
    Returns a weighted random choice from the normalized
    probability distribution of the top k most probable tokens,
    along with the probability of the chosen token.
    """
    topKIndices, topKValues = getTopKTokens(distribution, k, normalize=True)

    if topKValues.sum(dim=-1).to(torch.float32) == 0:
        raise ValueError("All values are 0.")

    # Sample a token index from the top-k indices using the probabilities
    sampled_idx = torch.multinomial(topKValues, num_samples=1).squeeze(-1)
    chosen_token_index = topKIndices.gather(dim=-1, index=sampled_idx.unsqueeze(-1)).squeeze(-1)
    chosen_token_prob = topKValues.gather(dim=-1, index=sampled_idx.unsqueeze(-1)).squeeze(-1)

    return chosen_token_index, chosen_token_prob
