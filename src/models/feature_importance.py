from fastai.tabular.all import *
import pandas as pd


class PermutationImportance:
    "Calculate and plot the permutation importance"

    def __init__(self, learn: Learner, df=None, bs=None):
        "Initialize with a test dataframe, a learner, and a metric"
        self.learn = learn
        self.df = df if df is not None else None
        bs = bs if bs is not None else learn.dls.bs
        self.dl = (
            learn.dls.test_dl(self.df, bs=bs) if self.df is not None else learn.dls[1]
        )
        self.x_names = learn.dls.x_names.filter(lambda x: "_na" not in x)
        self.na = learn.dls.x_names.filter(lambda x: "_na" in x)
        self.y = learn.dls.y_names
        self.results = self.calc_feat_importance()
        self.plot_importance(self.ord_dic_to_df(self.results))

    def measure_col(self, name: str):
        "Measures change after column shuffle"
        col = [name]
        if f"{name}_na" in self.na:
            col.append(name)
        orig = self.dl.items[col].values
        perm = np.random.permutation(len(orig))
        self.dl.items[col] = self.dl.items[col].values[perm]
        metric = self.learn.validate(dl=self.dl)[1]
        self.dl.items[col] = orig
        return metric

    def calc_feat_importance(self):
        "Calculates permutation importance by shuffling a column on a percentage scale"
        print("Getting base error")
        base_error = self.learn.validate(dl=self.dl)[1]
        self.importance = {}
        pbar = progress_bar(self.x_names)
        print("Calculating Permutation Importance")
        for col in pbar:
            self.importance[col] = self.measure_col(col)
        for key, value in self.importance.items():
            self.importance[key] = (
                base_error - value
            ) / base_error  # this can be adjusted
        return OrderedDict(
            sorted(self.importance.items(), key=lambda kv: kv[1], reverse=True)
        )

    def ord_dic_to_df(self, dict: OrderedDict):
        return pd.DataFrame(
            [[k, v] for k, v in dict.items()], columns=["feature", "importance"]
        )

    def plot_importance(self, df: pd.DataFrame, limit=20, asc=False, **kwargs):
        "Plot importance with an optional limit to how many variables shown"
        df_copy = df.copy()
        df_copy["feature"] = df_copy["feature"].str.slice(0, 25)
        df_copy = df_copy.sort_values(by="importance", ascending=asc)[
            :limit
        ].sort_values(by="importance", ascending=not (asc))
        ax = df_copy.plot.barh(x="feature", y="importance", **kwargs)
        for p in ax.patches:
            ax.annotate(
                f"{p.get_width():.4f}", ((p.get_width() * 1.005), p.get_y() * 1.005)
            )
        plt.show()
