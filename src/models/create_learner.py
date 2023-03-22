from fastai.tabular.all import *
import pandas as pd
import numpy as np
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
    )

    dls = to.dataloaders(bs=64)

    cardinalities = df[cat_names].nunique().to_numpy()
    emb_szs = {cat: min(50, card // 2) for cat, card in zip(cat_names, cardinalities)}

    config = tabular_config(y_range=[0, 120000], act_cls=nn.GELU())

    learn = tabular_learner(
        dls, metrics=mae, layers=[1000, 500, 250], emb_szs=emb_szs, config=config
    )

    if save_model_path:
        learn = learn.load(save_model_path, device=device)

    return learn


# function to embed features ,obtained from fastai forums
def embed_features(learner, df, device):
    df = df.copy()
    for i, feature in enumerate(learner.dls.cat_names):
        emb = learner.model.embeds[i]
        new_feat = pd.DataFrame(
            emb(tensor(df[feature], dtype=torch.int64, device=device)),
            index=df.index,
            columns=[f"{feature}_{j}" for j in range(emb.embedding_dim)],
        )
        df.drop(columns=feature, inplace=True)
        df = df.join(new_feat)
    return df
