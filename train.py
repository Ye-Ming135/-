import os
import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb

warnings.filterwarnings('ignore')


def load_data(data_dir):
    """加载训练数据和测试数据"""
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    sample_sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    return train, test, sample_sub


def build_features(df, categorical_cols):
    """特征工程"""
    df_new = df.copy()
    
    num_cols = [c for c in df.columns if c not in ['id', 'Irrigation_Need'] + categorical_cols]
    
    if len(num_cols) > 0:
        df_new['num_mean'] = df[num_cols].mean(axis=1)
        df_new['num_std'] = df[num_cols].std(axis=1)
        df_new['num_max'] = df[num_cols].max(axis=1)
        df_new['num_min'] = df[num_cols].min(axis=1)
        df_new['num_range'] = df_new['num_max'] - df_new['num_min']
    
    if 'Soil_Moisture' in df.columns and 'Temperature_C' in df.columns:
        df_new['moisture_temp_ratio'] = df['Soil_Moisture'] / (df['Temperature_C'] + 1)
    
    if 'Soil_pH' in df.columns and 'Organic_Carbon' in df.columns:
        df_new['ph_carbon_ratio'] = df['Soil_pH'] * df['Organic_Carbon']
    
    if 'Humidity' in df.columns and 'Temperature_C' in df.columns:
        df_new['heat_index'] = df['Humidity'] * df['Temperature_C']
    
    if 'Rainfall_mm' in df.columns and 'Sunlight_Hours' in df.columns:
        df_new['rain_sun_ratio'] = df['Rainfall_mm'] / (df['Sunlight_Hours'] + 1)
    
    if 'Previous_Irrigation_mm' in df.columns and 'Field_Area_hectare' in df.columns:
        df_new['irrigation_per_area'] = df['Previous_Irrigation_mm'] / (df['Field_Area_hectare'] + 0.1)
    
    return df_new


def encode_categorical(train_df, test_df, categorical_cols):
    """编码类别特征"""
    label_encoders = {}
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        train_encoded[col] = le.fit_transform(train_encoded[col].astype(str))
        
        test_vals = test_encoded[col].astype(str)
        mapping_dict = {k: v for k, v in zip(le.classes_, le.transform(le.classes_))}
        test_encoded[col] = test_vals.replace(mapping_dict).fillna(-1).astype(int)
        
        label_encoders[col] = le
    
    return train_encoded, test_encoded, label_encoders


def train_model(X, y, X_test, params, n_folds=5):
    """使用5折交叉验证训练LightGBM模型"""
    y_encoded = LabelEncoder().fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    params['num_class'] = num_classes
    
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_preds = np.zeros((len(X), num_classes))
    test_preds = np.zeros((len(X_test), num_classes))
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y_encoded)):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
        
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        train_set = lgb.Dataset(X_tr, label=y_tr)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
        
        model = lgb.train(
            params,
            train_set,
            valid_sets=[val_set],
            num_boost_round=2000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_folds
        
        val_pred_labels = np.argmax(oof_preds[val_idx], axis=1)
        fold_score = balanced_accuracy_score(y_val, val_pred_labels)
        fold_scores.append(fold_score)
        print(f"Fold {fold_idx + 1} Balanced Accuracy: {fold_score:.5f}")
    
    oof_pred_labels = np.argmax(oof_preds, axis=1)
    cv_score = balanced_accuracy_score(y_encoded, oof_pred_labels)
    
    return test_preds, cv_score, fold_scores, model


def generate_submission(test_preds, sample_sub, y):
    """生成提交文件"""
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    
    test_pred_labels = np.argmax(test_preds, axis=1)
    label_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
    
    submission = sample_sub.copy()
    submission['Irrigation_Need'] = [label_mapping[l] for l in test_pred_labels]
    
    return submission


def save_feature_importance(model, feature_cols, output_dir):
    """保存特征重要性"""
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    return feature_importance


def save_cv_result(cv_score, fold_scores, output_dir):
    """保存交叉验证结果"""
    cv_result = pd.DataFrame({
        'metric': ['balanced_accuracy'],
        'score': [cv_score],
        'mean_fold_score': [np.mean(fold_scores)],
        'std_fold_score': [np.std(fold_scores)],
        'fold_scores': [str([round(s, 5) for s in fold_scores])]
    })
    
    cv_result.to_csv(os.path.join(output_dir, 'cv_result.csv'), index=False)
    return cv_result


def main(args):
    print("=" * 60)
    print("Kaggle Playground Series S6E4 - Irrigation Need Prediction")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n[1/6] 加载数据...")
    train, test, sample_sub = load_data(args.data_dir)
    print(f"训练集: {train.shape}")
    print(f"测试集: {test.shape}")
    print(f"目标分布:\n{train['Irrigation_Need'].value_counts()}")
    
    categorical_cols = [c for c in train.select_dtypes(include=['object']).columns.tolist() 
                        if c != 'Irrigation_Need']
    numerical_cols = [c for c in train.select_dtypes(include=['int64', 'float64']).columns.tolist()
                      if c not in ['id', 'Irrigation_Need']]
    
    print(f"\n类别特征 ({len(categorical_cols)}): {categorical_cols}")
    print(f"数值特征 ({len(numerical_cols)}): {numerical_cols}")
    
    print("\n[2/6] 特征工程...")
    train_fe = build_features(train, categorical_cols)
    test_fe = build_features(test, categorical_cols)
    print(f"特征工程后: {train_fe.shape[1]} 个特征")
    
    print("\n[3/6] 编码类别特征...")
    train_encoded, test_encoded, _ = encode_categorical(train_fe, test_fe, categorical_cols)
    
    feature_cols = [c for c in train_encoded.columns if c not in ['id', 'Irrigation_Need']]
    X = train_encoded[feature_cols]
    y = train_encoded['Irrigation_Need']
    X_test = test_encoded[feature_cols]
    
    print("\n[4/6] 训练模型...")
    lgb_params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "num_leaves": args.num_leaves,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "bagging_fraction": args.bagging_fraction,
        "feature_fraction": args.feature_fraction,
        "random_state": args.seed,
        "verbose": -1,
        "num_threads": args.num_threads
    }
    
    test_preds, cv_score, fold_scores, model = train_model(
        X, y, X_test, lgb_params, n_folds=args.n_folds
    )
    
    print("\n[5/6] 保存结果...")
    feature_importance = save_feature_importance(model, feature_cols, args.output_dir)
    cv_result = save_cv_result(cv_score, fold_scores, args.output_dir)
    
    print(f"\n各折平衡准确率: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"平均平衡准确率: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
    print(f"整体OOF平衡准确率: {cv_score:.5f}")
    print(f"\n特征重要性(Top 10):\n{feature_importance.head(10)}")
    
    print("\n[6/6] 生成提交文件...")
    submission = generate_submission(test_preds, sample_sub, y)
    submission_path = os.path.join(args.output_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"提交文件已保存: {submission_path}")
    print(f"预测分布:\n{submission['Irrigation_Need'].value_counts()}")
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kaggle Irrigation Need Prediction')
    
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录 (default: ./data)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录 (default: ./output)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='交叉验证折数 (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (default: 42)')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='线程数 (default: 4)')
    
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='学习率 (default: 0.05)')
    parser.add_argument('--max_depth', type=int, default=8,
                        help='最大深度 (default: 8)')
    parser.add_argument('--num_leaves', type=int, default=63,
                        help='叶子节点数 (default: 63)')
    parser.add_argument('--reg_alpha', type=float, default=0.1,
                        help='L1正则化 (default: 0.1)')
    parser.add_argument('--reg_lambda', type=float, default=0.1,
                        help='L2正则化 (default: 0.1)')
    parser.add_argument('--bagging_fraction', type=float, default=0.8,
                        help='采样比例 (default: 0.8)')
    parser.add_argument('--feature_fraction', type=float, default=0.8,
                        help='特征采样比例 (default: 0.8)')
    
    args = parser.parse_args()
    main(args)
