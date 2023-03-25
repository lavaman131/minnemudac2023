from fastai.tabular.all import *
import pandas as pd
import numpy as np
import torch
from torch import nn
from typing import List, Union
from fastai.losses import BaseLoss


class RMSELoss(BaseLoss):
    "Root Mean Squared Error loss"

    def __init__(self, *args, axis=-1, floatify=True, **kwargs):
        super().__init__(
            nn.MSELoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs
        )

    def __call__(self, inp, targ, **kwargs):
        return super().__call__(torch.sqrt(inp), torch.sqrt(targ), **kwargs)


def embedding_rule(n_cats: int):
    return min(50, n_cats // 2)


def create_model(
    dls: TabularDataLoaders,
    metrics,
    loss_func,
    cat_names: List[str],
    cont_names: List[str],
    training: bool = True,
    save_model_path: Union[None, str] = None,
    device: str = None,
) -> Learner:

    emb_szs = [
        (
            len(np.unique(dls.train_ds[cat])),
            embedding_rule(len(np.unique(dls.train_ds[cat]))),
        )
        for cat in cat_names
    ]

    model = TabularModel(
        emb_szs=emb_szs,
        n_cont=len(cont_names),
        out_sz=1,
        layers=[1000, 500, 250],
        y_range=[0, 120000],
        act_cls=nn.GELU(),
    ).to(device)

    learn = TabularLearner(dls, model, metrics=metrics, loss_func=loss_func)

    if not training:
        if save_model_path:
            learn = learn.load(save_model_path, device=device)
        else:
            raise ValueError("Must specify model path if not training.")

    return learn


# function to embed features ,obtained from fastai forums
def embed_features(learner, df, cat_names, device):
    df = df.copy()
    for i, feature in enumerate(cat_names):
        emb = learner.embeds[i]
        new_feat = pd.DataFrame(
            emb(tensor(df[feature], dtype=torch.int64, device=device)),
            index=df.index,
            columns=[f"{feature}_{j}" for j in range(emb.embedding_dim)],
        )
        df.drop(columns=feature, inplace=True)
        df = df.join(new_feat)
    return df