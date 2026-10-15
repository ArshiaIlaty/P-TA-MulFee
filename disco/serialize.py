"""
DISCO-TAB: schema-aware linearization L: row -> text, and its inverse.

Deterministic, shared across datasets (Section IV-A of the paper):
  - fixed column order (features first, target last);
  - numeric values standardized by REAL-TRAIN mean/std, rendered with fixed
    decimal precision;
  - categoricals as canonical string identifiers;
  - missing values encoded with the reserved token [MISSING].

The forward pass also returns, per column, the character span of the *value*
substring, so the feature-level discriminator can localize value tokens.
"""
import re
import numpy as np
import pandas as pd

MISSING = "[MISSING]"


class Serializer:
    def __init__(self, num_cols, cat_cols, target="target", num_decimals=2):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.target = target
        # Generation order: anchor on the first feature (the generation prompt),
        # then the TARGET, then the remaining features. Emitting the target early
        # guarantees it is captured even if the LM later degenerates/loops on a
        # high-dimensional schema, and makes feature generation class-conditional.
        feats = self.num_cols + self.cat_cols
        self.order = ([feats[0], self.target] + feats[1:]) if feats else [self.target]
        self.num_decimals = num_decimals

    def _build_display_names(self):
        # Alias "hard" feature names (special chars, long, or near-duplicates)
        # to simple `featN` tokens the LM can enumerate reliably; keep simple
        # names (and the target) for semantic signal.
        feats = self.num_cols + self.cat_cols
        simple = re.compile(r"^[A-Za-z0-9_ ]{1,25}$")
        # "Confusable" family: >=3 names that are identical after stripping
        # trailing digits (e.g. Attribute1..Attribute20) -- the LM mis-enumerates
        # these, so alias the whole family.
        import collections
        stem = lambda s: re.sub(r"\d+$", "", str(s))
        stem_counts = collections.Counter(stem(c) for c in feats)
        confusable = {c for c in feats if stem_counts[stem(c)] >= 3}
        self.display = {}
        for i, c in enumerate(feats):
            ok = simple.match(str(c)) and c not in confusable
            self.display[c] = str(c) if ok else f"feat{i}"
        # Enforce uniqueness of display names.
        seen = {}
        for i, c in enumerate(feats):
            d = self.display[c]
            if d in seen:
                self.display[c] = f"feat{i}"
            seen[self.display[c]] = c
        self.display[self.target] = str(self.target)
        self.rev = {d: c for c, d in self.display.items()}

    def disp(self, col):
        return self.display.get(col, str(col))

    def fit(self, df):
        self._build_display_names()
        self.mean_, self.std_, self.round_ = {}, {}, {}
        for c in self.num_cols:
            v = pd.to_numeric(df[c], errors="coerce")
            m, s = float(v.mean()), float(v.std())
            self.mean_[c] = m
            self.std_[c] = s if s > 1e-8 else 1.0
            # Infer display precision from the real data (0 => integer-valued).
            frac = v.dropna() % 1
            self.round_[c] = 0 if (frac.abs() < 1e-9).all() else 4
        # Canonical categorical vocabularies (for validation on parse).
        self.cat_vocab_ = {c: set(df[c].astype(str)) for c in self.cat_cols}
        self.cat_vocab_[self.target] = set(df[self.target].astype(str))
        return self

    # ---- forward ----
    def _fmt_value(self, col, val):
        if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
            return MISSING
        if col in self.num_cols:
            try:
                z = (float(val) - self.mean_[col]) / self.std_[col]
            except (ValueError, TypeError):
                return MISSING
            return f"{z:.{self.num_decimals}f}"
        return str(val).strip()

    def serialize_row(self, row, with_spans=False):
        parts, spans, pos = [], {}, 0
        for i, col in enumerate(self.order):
            prefix = f"{self.disp(col)} is "
            val = self._fmt_value(col, row[col] if col in row else np.nan)
            clause = prefix + val
            sep = "" if i == 0 else ", "
            start = pos + len(sep) + len(prefix)
            end = start + len(val)
            spans[col] = (start, end)
            parts.append(sep + clause)
            pos = end
        text = "".join(parts)
        return (text, spans) if with_spans else text

    def serialize_df(self, df):
        return [self.serialize_row(df.iloc[i]) for i in range(len(df))]

    def clause_texts(self, row):
        """List of per-feature "col is val" clauses (for the sentence disc)."""
        return [f"{self.disp(c)} is {self._fmt_value(c, row[c] if c in row else np.nan)}"
                for c in self.order]

    # ---- inverse ----
    def _unfmt_value(self, col, token):
        token = token.strip()
        if token == MISSING or token == "":
            return np.nan
        if col in self.num_cols:
            try:
                z = float(token)
            except ValueError:
                return np.nan
            raw = z * self.std_[col] + self.mean_[col]
            r = self.round_[col]
            return int(round(raw)) if r == 0 else round(raw, r)
        return token

    def value_spans(self, text):
        """Char spans of each column's VALUE in an arbitrary serialized string.

        Works on both real and generated text (deterministic format), so the
        feature discriminator can localize value tokens the same way for both.
        """
        spans, pos = {}, 0
        segs = text.split(", ")
        for i, seg in enumerate(segs):
            sep = 0 if i == 0 else 2  # len(", ")
            seg_start = pos + sep
            if " is " in seg:
                key, _, val = seg.partition(" is ")
                real = self.rev.get(key.strip(), key.strip())
                if real in self.order:
                    vstart = seg_start + len(key) + len(" is ")
                    spans[real] = (vstart, vstart + len(val))
            pos = seg_start + len(seg)
        return spans

    def parse_text(self, text):
        """Parse one serialized record back to a dict; None if unusable."""
        out = {}
        for clause in text.split(","):
            if " is " not in clause:
                continue
            key, _, val = clause.partition(" is ")
            real = self.rev.get(key.strip(), key.strip())
            if real in self.order:
                out[real] = self._unfmt_value(real, val)
        return out

    def parse_batch(self, texts, require_target=True, min_feat_frac=0.5):
        """Keep a generated record if it has a valid target and at least
        `min_feat_frac` of the feature columns; missing features -> NaN (handled
        downstream by the preprocessor). This tolerates the LM occasionally
        dropping a clause on high-dimensional/repetitively-named schemas."""
        rows = []
        feat_cols = self.num_cols + self.cat_cols
        need = max(1, int(np.ceil(min_feat_frac * len(feat_cols))))
        for t in texts:
            d = self.parse_text(t)
            if require_target and (self.target not in d or pd.isna(d.get(self.target))):
                continue
            if require_target and str(d.get(self.target)) not in self.cat_vocab_[self.target]:
                continue
            present = sum(1 for c in feat_cols if c in d and not pd.isna(d.get(c)))
            if present < need:
                continue
            rows.append(d)
        if not rows:
            return pd.DataFrame(columns=self.order)
        df = pd.DataFrame(rows)
        for c in self.order:
            if c not in df.columns:
                df[c] = np.nan
        return df[self.order]


if __name__ == "__main__":
    from data import load_dataset
    for name in ["heart_failure", "obesity"]:
        b = load_dataset(name, seed=0)
        s = Serializer(b.num_cols, b.cat_cols, b.target).fit(b.df_train)
        txt, spans = s.serialize_row(b.df_train.iloc[0], with_spans=True)
        print(f"\n=== {name} ===")
        print("TEXT:", txt[:200])
        # verify span extraction
        c0 = b.feature_cols[0]
        st, en = spans[c0]
        print(f"span[{c0}] -> '{txt[st:en]}'")
        back = s.parse_batch([txt])
        print("ROUNDTRIP cols ok:", list(back.columns) == s.order, "rows:", len(back))
        if len(back):
            print("recovered target:", back[b.target].iloc[0],
                  "| orig:", b.df_train.iloc[0][b.target])
