import os, argparse, warnings 
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

def load_data(data_dir):
    # 加载数据
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    return train, test, sub


def build_features(df, cat_cols):
    """特征工程"""
    df_new = df.copy()

    # 数值特征统计
    num_cols = [c for c in df.columns if c not in ['id', 'Irrigation_Need'] + cat_cols]
    if len(num_cols) > 0:
        df_new['num_mean'] = df[num_cols].mean(axis=1)
        df_new['num_std'] = df[num_cols].std(axis=1)
        df_new['num_max'] = df[num_cols].max(axis=1)
        df_new['num_min'] = df[num_cols].min(axis=1)
        df_new['num_range'] = df_new['num_max'] - df_new['num_min']

    # 交叉特征
    if 'Soil_Moisture' in df.columns and 'Temperature_C' in df.columns:
        df_new['moisture_temp_ratio'] = df['Soil_Moisture'] / (df['Temperature_C'] + 1)

    if 'Soil_pH' in df.columns and 'Organic_Carbon' in df.columns:
        df_new['ph_carbon_ratio'] = df['Soil_pH'] * df['Organic_Carbon']

    if 'Humidity' in df.columns and 'Temperature_C' in df.columns:
        df_new['heat_idx'] = df['Humidity'] * df['Temperature_C']

    if 'Rainfall_mm' in df.columns and 'Sunlight_Hours' in df.columns:
        df_new['rain_sun'] = df['Rainfall_mm'] / (df['Sunlight_Hours'] + 1)

    if 'Previous_Irrigation_mm' in df.columns and 'Field_Area_hectare' in df.columns:
        df_new['irr_per_area'] = df['Previous_Irrigation_mm'] / (df['Field_Area_hectare'] + 0.1)

    return df_new


def encode_cat(train_df, test_df, cat_cols):
    """类别特征编码"""
    encoders = {}
    tr = train_df.copy()
    te = test_df.copy()

    for col in cat_cols:
        le = LabelEncoder()
        tr[col] = le.fit_transform(tr[col].astype(str))
        mapping = {k: v for k, v in zip(le.classes_, le.transform(le.classes_))}
        te[col] = te[col].astype(str).replace(mapping).fillna(-1).astype(int)
        encoders[col] = le

    return tr, te, encoders


# 训练函数
def train_cv(X, y, X_test, params, nfolds=5):
    y_enc = LabelEncoder().fit_transform(y)
    n_cls = len(np.unique(y_enc))
    params['num_class'] = n_cls

    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
    oof = np.zeros((len(X), n_cls))
    pred_t = np.zeros((len(X_test), n_cls))
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_enc)):
        print(f"\n--- Fold {fold + 1}/{nfolds} ---")
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y_enc[tr_idx], y_enc[va_idx]

        dtrain = lgb.Dataset(Xtr, label=ytr)
        dval = lgb.Dataset(Xva, label=yva, reference=dtrain)

        mdl = lgb.train(
            params, dtrain,
            valid_sets=[dval],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )

        oof[va_idx] = mdl.predict(Xva, num_iteration=mdl.best_iteration)
        pred_t += mdl.predict(X_test, num_iteration=mdl.best_iteration) / nfolds

        val_preds = np.argmax(oof[va_idx], axis=1)
        sc = balanced_accuracy_score(yva, val_preds)
        scores.append(sc)
        print(f"Fold {fold + 1} score: {sc:.5f}")

    oof_labels = np.argmax(oof, axis=1)
    cv_sc = balanced_accuracy_score(y_enc, oof_labels)

    return pred_t, cv_sc, scores, mdl


def make_submission(preds, sample, y):
    """生成提交"""
    le = LabelEncoder()
    le.fit(y)
    labels = np.argmax(preds, axis=1)
    mapping = dict(zip(range(len(le.classes_)), le.classes_))

    sub = sample.copy()
    sub['Irrigation_Need'] = [mapping[l] for l in labels]
    return sub


def save_stuff(model, feat_cols, cv_sc, fold_scores, out_dir):
    """保存特征重要性和cv结果"""
    # 特征重要性
    imp = pd.DataFrame({
        'feature': feat_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    imp.to_csv(os.path.join(out_dir, 'feature_importance.csv'), index=False)

    # cv结果
    cv_df = pd.DataFrame({
        'metric': ['balanced_accuracy'],
        'score': [cv_sc],
        'mean_fold': [np.mean(fold_scores)],
        'std_fold': [np.std(fold_scores)],
        'folds': [str([round(s, 5) for s in fold_scores])]
    })
    cv_df.to_csv(os.path.join(out_dir, 'cv_result.csv'), index=False)

    return imp, cv_df


def main(args):
    print("=" * 50)
    print("Kaggle S6E4 - Irrigation Prediction")
    print("=" * 50)

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[1] 加载数据..")
    train, test, sample = load_data(args.data_dir)
    print(f"train: {train.shape}, test: {test.shape}")
    print(train['Irrigation_Need'].value_counts())

    cat_cols = [c for c in train.select_dtypes(include=['object']).columns if c != 'Irrigation_Need']
    num_cols = [c for c in train.select_dtypes(include=['int64', 'float64']).columns if
                c not in ['id', 'Irrigation_Need']]
    print(f"类别特征: {len(cat_cols)}个, 数值特征: {len(num_cols)}个")

    print("\n[2] 特征工程..")
    tr_feat = build_features(train, cat_cols)
    te_feat = build_features(test, cat_cols)
    print(f"处理后 {tr_feat.shape[1]} 个特征")

    print("\n[3] 编码..")
    tr_enc, te_enc, _ = encode_cat(tr_feat, te_feat, cat_cols)
    feat_cols = [c for c in tr_enc.columns if c not in ['id', 'Irrigation_Need']]

    X = tr_enc[feat_cols]
    y = tr_enc['Irrigation_Need']
    Xt = te_enc[feat_cols]

    print("\n[4] 训练..")
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": args.lr,
        "max_depth": args.depth,
        "num_leaves": args.leaves,
        "reg_alpha": args.alpha,
        "reg_lambda": args.lam,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "random_state": args.seed,
        "verbose": -1,
        "num_threads": args.njobs
    }

    preds, cv_sc, fold_sc, model = train_cv(X, y, Xt, params, args.nfolds)

    print("\n[5] 保存..")
    imp, _ = save_stuff(model, feat_cols, cv_sc, fold_sc, args.output_dir)
    print(f"各折: {[f'{s:.5f}' for s in fold_sc]}")
    print(f"mean: {np.mean(fold_sc):.5f} (+/-{np.std(fold_sc):.5f})")
    print(f"OOF: {cv_sc:.5f}")
    print(f"Top10特征:\n{imp.head(10)}")

    print("\n[6] 提交文件..")
    sub = make_submission(preds, sample, y)
    out_path = os.path.join(args.output_dir, 'submission.csv')
    sub.to_csv(out_path, index=False)
    print(f"saved: {out_path}")
    print(sub['Irrigation_Need'].value_counts())
    print("\nDone!")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='./data')
    p.add_argument('--output_dir', type=str, default='./output')
    p.add_argument('--nfolds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--njobs', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--depth', type=int, default=8)
    p.add_argument('--leaves', type=int, default=63)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--lam', type=float, default=0.1)
    args = p.parse_args()
    main(args)
