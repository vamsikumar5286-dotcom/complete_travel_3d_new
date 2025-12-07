import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Reduce cardinality by mapping rare categories to 'OTHER' and convert to numeric codes."""
    def __init__(self, cols=None, top_k=200):
        self.cols = cols or []
        self.top_k = top_k
        self.mappings = {}

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for c in self.cols:
            vals = df[c].astype(str).fillna('NA')
            top = vals.value_counts().nlargest(self.top_k).index.tolist()
            self.mappings[c] = set(top)
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for c in self.cols:
            df[c] = df[c].astype(str).fillna('NA')
            top_set = self.mappings.get(c, set())
            df[c] = df[c].where(df[c].isin(top_set), other='OTHER')
            df[c] = pd.Categorical(df[c]).codes.astype(np.int32)
        return df


class WrapperPipeline:
    """Wrapper pipeline that applies FrequencyEncoder then the model."""
    def __init__(self, encoder, model, feature_cols, numeric_feats):
        self.encoder = encoder
        self.model = model
        self.feature_cols = feature_cols
        self.numeric_feats = numeric_feats

    def predict(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        for c in self.feature_cols:
            if c not in df.columns:
                df[c] = 0
        # Identify categorical columns as those in feature_cols but not in numeric_feats
        cat_cols = [c for c in self.feature_cols if c not in self.numeric_feats]

        # Ensure numeric cols exist and are numeric
        for n in self.numeric_feats:
            if n not in df.columns:
                df[n] = 0
        num_df = df[self.numeric_feats].astype(float).fillna(0.0).reset_index(drop=True)

        # Transform categorical columns using encoder (produces TE columns)
        te_df = self.encoder.transform(df[cat_cols]).reset_index(drop=True)

        # Concatenate numeric features then TE features to match training order
        X_pred = pd.concat([num_df, te_df], axis=1)
        return self.model.predict(X_pred)


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Simple target encoder with smoothing; can be fit on full data to transform new rows.

    Usage:
      te = TargetEncoder(cols=['Source','Destination'], m=10)
      te.fit(X, y)
      X_enc = te.transform(X)
    """
    def __init__(self, cols=None, m=10):
        self.cols = cols or []
        self.m = m
        self.maps = {}
        self.prior = None

    def fit(self, X, y):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y = pd.Series(y).astype(float)
        self.prior = y.mean()
        for c in self.cols:
            vals = df[c].astype(str).fillna('NA')
            grp = pd.DataFrame({'k': vals, 'y': y})
            agg = grp.groupby('k')['y'].agg(['sum', 'count'])
            # smoothing
            agg['enc'] = (agg['sum'] + self.m * self.prior) / (agg['count'] + self.m)
            self.maps[c] = agg['enc'].to_dict()
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        out = pd.DataFrame(index=df.index)
        for c in self.cols:
            series = df[c].astype(str).fillna('NA')
            mapping = self.maps.get(c, {})
            out[c + '_te'] = series.map(mapping).fillna(self.prior).astype(float)
        return out
