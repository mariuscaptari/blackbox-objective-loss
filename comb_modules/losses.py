import torch


class HammingLoss(torch.nn.Module):
    def forward(self, suggested, target):
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()
        # return (torch.mean(suggested*(1.0-target)) + torch.mean((1.0-suggested)*target)) * 25.0


class PathCostLoss(torch.nn.Module):
    """
    Loss function that computes the sum of true distances/costs for the suggested path.
    This is the dot product of the suggested binary path with real costs.
    """
    def forward(self, suggested_path, true_costs):
        """
        Args:
            suggested_path: Binary tensor indicating which vertices are part of the suggested path
            true_costs: Real costs/distances for each vertex
        Returns:
            Sum of true costs for the suggested path
        """
        # Element-wise multiplication gives cost at each vertex if it's in the path
        path_costs = suggested_path * true_costs
        # Sum across all dimensions except batch dimension
        return path_costs.sum(dim=tuple(range(1, len(path_costs.shape))))


#
