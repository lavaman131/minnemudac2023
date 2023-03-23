from fastai.tabular.all import *
import pandas as pd
import numpy as np
import torch
from torch import nn
from typing import List, Union


def init_model(
    df: pd.DataFrame,
    cat_names: List[str],
    cont_names: List[str],
    y_names: str,
    save_model_path: Union[None, str] = None,
    device: str = None,
) -> Learner:
    splits = RandomSplitter(valid_pct=0.2)(range_of(df))

    to = TabularPandas(
        df,
        procs=[Categorify, FillMissing, Normalize],
        cat_names=cat_names,
        cont_names=cont_names,
        y_names=y_names,
        splits=splits,
        device=device
    )

    dls = to.dataloaders(bs=64)

    emb_szs = [(43, 21), (28, 13), (8, 3), (31, 15), (3, 1), (31, 15), (3, 1)]

    model = TabularModel(
        emb_szs=emb_szs,
        n_cont=len(cont_names),
        out_sz=1,
        layers=[1000, 500, 250],
        y_range=[0, 120000],
        act_cls=nn.GELU(),
    ).to(device)

    learn = TabularLearner(dls, model, metrics=mae)

    if save_model_path:
        learn = learn.load(save_model_path, device=device).eval()

    return learn, to


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
