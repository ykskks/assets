oof = np.zeros(len(X_train))
predictions = np.zeros(len(X_test))
feature_importance_df = pd.DataFrame()
fold_scores = []

# for fold, (train_idx, val_idx) in enumerate(folds.split(X_train, groups=groups)):
for fold, (train_idx, val_idx) in enumerate(folds.split(X_train, y_to_stratify)):
    self.logger.debug("-" * 100)
    self.logger.debug(f"Fold {fold+1}")
    train_data = lgb.Dataset(X_train.iloc[train_idx], label=y_train.iloc[train_idx])
    val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_train.iloc[val_idx])
    callbacks = [log_evaluation(self.logger, period=100)]
    clf = lgb.train(self.params, train_data, valid_sets=[train_data, val_data], verbose_eval=100, early_stopping_rounds=100, callbacks=callbacks)  #, feval=eval_func)
    oof[val_idx] = clf.predict(X_train.iloc[val_idx].values, num_iteration=clf.best_iteration)
    fold_score = mean_squared_log_error(np.expm1(y_train.iloc[val_idx].values), np.expm1(oof[val_idx])) ** .5
    fold_scores.append(fold_score)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = X_train.columns.values
    fold_importance_df["importance"] = clf.feature_importance(importance_type="gain")
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += np.expm1(clf.predict(X_test, num_iteration=clf.best_iteration)) / folds.n_splits

_feature_importance_df = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)  # .head(50)
self.logger.debug("##### feature importance #####")
self.logger.debug(_feature_importance_df.head(50))
cv_score_fold_mean = sum(fold_scores) / len(fold_scores)
self.logger.debug(f"cv_score_fold_mean: {cv_score_fold_mean}")