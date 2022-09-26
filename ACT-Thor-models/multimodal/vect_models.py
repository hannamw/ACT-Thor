import json
import os.path
from collections import OrderedDict

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple


__allowed_conditioning_methods__ = ['concat', 'embedding']
__allowed_activations__ = {
    'none': nn.Identity,
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}

__shared_embedding_size__ = 256
__action_embedding_size__ = 768


class AbstractVectorTransform(ABC, nn.Module):
    """
    Base class for all models of Vector Transformation (VecT).

    This is needed in order to have a general structure for embedding actions and processing batches.
    """

    @abstractmethod
    def __init__(self, actions: set, vec_size: int,
                 device='cpu', **kwargs):
        super(AbstractVectorTransform, self).__init__()
        self.actions = actions
        self.nr_actions = len(self.actions)
        self.vec_size = vec_size
        self.device = device

    @abstractmethod
    def forward(self, before, action, after, negatives):
        pass

    def process_batch(self, batch, loss_fn=None):
        before = batch['before'].to(self.device)
        actions = batch['action'].to(self.device).long()
        after = batch['positive'].to(self.device)
        negatives = batch['negatives'].to(self.device)

        before, predictions, after, negatives = self(before, actions, after, negatives)

        if self.training and loss_fn is not None:
            return loss_fn(predictions, after)
        elif not self.training and loss_fn is None:
            if len(after.shape) == 2:
                after.unsqueeze_(0)  # adding fake contrast dimension (for computing distance)
            return predictions, after

    def to(self, *args, **kwargs):
        self.device = args[0]
        res = super(AbstractVectorTransform, self).to(*args, **kwargs)
        return res

    def setup_embedding_layers(self, hidden_sizes):
        layer_sizes = [self.vec_size, self.vec_size]

        if self.conditioning_method == 'concat':
            layer_sizes[0] += self.nr_actions

            self.__eye_helper = torch.eye(self.nr_actions, device=self.device)

            def embed_action(action_batch: torch.LongTensor) -> torch.Tensor:
                return torch.stack([self.__eye_helper[i].detach().clone() for i in action_batch], dim=0).to(self.device)

            self.embed_visual = nn.Identity()
            self.embed_action = embed_action

        elif self.conditioning_method == 'embedding':
            layer_sizes[0] = 2 * __shared_embedding_size__
            layer_sizes[-1] = __shared_embedding_size__
            self.embed_visual = nn.Linear(self.vec_size, __shared_embedding_size__)
            self.embed_action = nn.Sequential(
                nn.Embedding(self.nr_actions, __action_embedding_size__),
                nn.Linear(__action_embedding_size__, __shared_embedding_size__)
            )

        if hidden_sizes is None:
            h = int(layer_sizes[0] * 0.75)
            hidden_sizes = [h]
        elif hidden_sizes == 'large':
            h = int(layer_sizes[0] * 0.75)
            hidden_sizes = [h, h // 2, h]
        self.hidden_sizes = hidden_sizes

        layer_sizes = [layer_sizes[0]] + hidden_sizes + [layer_sizes[-1]]

        return layer_sizes


__supported_regressions__ = {None, 'torch'}


class LinearVectorTransform(AbstractVectorTransform):
    """Module implementing vector transformation as matrix multiplication. Can also be initialized with a least-squares
    regression procedure.

    In the main paper, this is called Action-Matrix (AM).

    """

    # TODO implement this as a single tensor to allow batched learning
    def __init__(self, actions: set, vec_size: int,
                 use_regression=False,
                 regression_type=None,
                 **kwargs):
        super(LinearVectorTransform, self).__init__(actions, vec_size, **kwargs)
        self.regression_type = regression_type
        assert self.regression_type in __supported_regressions__

        self.use_regression = use_regression

        self.weights = nn.Parameter(torch.stack([nn.Linear(self.vec_size, self.vec_size, bias=False).weight.clone().detach() for i in range(self.nr_actions)], dim=0))


    def forward(self, before, action, after=None, negatives=None):
        # Normalize vector?
        pred = before.unsqueeze(1).bmm(self.weights.index_select(0, action))
        return before, pred.squeeze(1), after, negatives

    def regression_init(self, data: Tuple, regression_type=None):
        # data is a tuple of dicts ({'action' --> samples matrix A}, {'action' --> target vector B})
        # A, B: should be (num. samples, vec_size)
        if self.regression_type is None and regression_type is None:
            raise RuntimeError("unspecified modality of regression for initialization")

        if self.regression_type is None:
            self.regression_type = regression_type
            assert self.regression_type in __supported_regressions__

        train_mat, target_mat = data

        if self.regression_type == 'torch':
            sorted_actions = sorted(train_mat.keys(), key=lambda el: int(el))  # sort action indices
            weights = []
            for action in sorted_actions:
                sol = torch.linalg.lstsq(train_mat[action], target_mat[action]).solution.to(self.device)
                if sol.numel() == 0:
                    sol = torch.eye(self.vec_size, device=self.device)
                weights.append(sol)
            self.weights = nn.Parameter(torch.stack(weights, dim=0).to(self.device))
        else:
            pass  # TODO

        # Freezes parameters after performing regression
        for p in self.parameters():
            p.requires_grad_(False)

    # Use numpy or sklearn to solve partial least squares?
    # Separate initialization (solution of lsqs) and parameter assignment


class LinearConcatVecT(AbstractVectorTransform):
    """
    Implements vector transformation as concatenation of visual embedding and action embedding followed
    by a multiplication by a unique weight matrix.

    In the main paper, this is called Concat-Linear (CL).
    """

    def __init__(self, actions: set, vec_size: int, conditioning_method='concat', **kwargs):
        super().__init__(actions, vec_size, **kwargs)

        self.conditioning_method = conditioning_method

        layer_sizes = self.setup_embedding_layers([])

        self.net = nn.Linear(layer_sizes[0], layer_sizes[-1])

    def forward(self, before, action, after=None, negatives=None):
        before = self.embed_visual(before)
        pred = self.net(torch.cat([before, self.embed_action(action)], dim=-1))
        if after is not None:
            after = self.embed_visual(after)
        if negatives is not None:
            negatives = self.embed_visual(negatives)
        return before, pred, after, negatives


class ConditionalFCNVecT(AbstractVectorTransform):
    """
    Module implementing vector transform as a conditional feedforward network, where the input
    vector is combined with the action label embedding to provide conditionality.

    In the main paper it is named Concat-Multi (CM).
    """

    def __init__(self, actions: set, vec_size: int,
                 hidden_sizes=None,
                 dropout_p: float = 0.0,
                 use_bn: bool = False,
                 activation='none',
                 conditioning_method='concat',
                 **kwargs):
        super(ConditionalFCNVecT, self).__init__(actions, vec_size)
        self.dropout_p = dropout_p
        self.use_bn = use_bn

        assert activation in __allowed_activations__
        self.activation = activation

        assert conditioning_method in __allowed_conditioning_methods__
        self.conditioning_method = conditioning_method

        layer_sizes = self.setup_embedding_layers(hidden_sizes)

        # TODO: check ordering of BatchNorm and activation
        self.net = nn.Sequential(*[
                nn.Sequential(
                    OrderedDict([
                        ('dropout', nn.Dropout(self.dropout_p)),
                        ('linear', nn.Linear(layer_sizes[i], layer_sizes[i + 1])),
                        ('bnorm', nn.BatchNorm1d(layer_sizes[i + 1]) if (self.use_bn and i < len(layer_sizes) - 2) else nn.Identity()),  # up to penultimate layer
                        ('activation', __allowed_activations__[self.activation]() if i < len(layer_sizes) - 2 else nn.Identity())  # up to penultimate layer
                    ])
                )
                for i in range(len(layer_sizes) - 1)]  # because building with (i, i+1)
            )

    def forward(self, before, action, after=None, negatives=None):
        before = self.embed_visual(before)
        pred = self.net(torch.cat([before, self.embed_action(action)], dim=-1))
        if after is not None:
            after = self.embed_visual(after)
        if negatives is not None:
            negatives = self.embed_visual(negatives)
        return before, pred, after, negatives


__name_map__ = {
    'Action-Matrix': LinearVectorTransform,
    'Concat-Linear': LinearConcatVecT,
    'Concat-Multi': ConditionalFCNVecT
}


def load_model(path, actions=None, data_folder="new-dataset/data-improved-descriptions"):
    """
    Loads a VectorTransform model with the predefined set of actions.

    If not defined, actions are loaded from the specified data folder. The dataset should automatically save actions
    in the same folder the first time is loaded.
    """
    if actions is None:
        with open(os.path.join(data_folder, "actions.json"), mode='rt') as fp:
            actions = set(json.load(fp).keys())

    sd = torch.load(path)
    ks = list(sd.keys())

    print(ks)
    if len(ks) == 1:
        vec_size = sd['weights'][0].shape[-1]
        model = LinearVectorTransform(actions, vec_size)
    else:
        if any(['embed' in k for k in ks]):
            conditioning_method = 'embedding'
            vec_size = sd['embed_visual.weight'].shape[-1]
        else:
            conditioning_method = 'concat'
            vec_size = [el for k, el in sd.items() if ('net' in k) and ('weight' in k)][0].shape[-1] - len(actions)

        use_bnorm = True
        activation = 'relu'

        if len([k for k in ks if 'net' in k]) == 2:
            model = LinearConcatVecT(actions, vec_size, conditioning_method)
        else:
            model = ConditionalFCNVecT(actions, vec_size, use_bn=use_bnorm, activation=activation, conditioning_method=conditioning_method)

    model.load_state_dict(sd)
    return model
