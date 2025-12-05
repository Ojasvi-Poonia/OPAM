import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import joblib
import os


class OPAMExpensePredictor:
    """AI/ML Model for predicting next month's expenses"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.category_models = {}
        self.tuned_params = {}  # NEW: Store tuned parameters
        self.tuning_results = {}  # NEW: Store tuning comparison results
        
    def print_header(self, text, char="="):
        """Print formatted header"""
        print(f"\n{char * 80}")
        print(f"  {text}")
        print(f"{char * 80}\n")
    
    def load_and_prepare_data(self, df):
        """Load and prepare transaction data"""
        self.print_header("📊 DATA LOADING & PREPARATION")
        
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"✓ Total Transactions: {len(df):,}")
        print(f"✓ Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"✓ Total Spending: ₹{df['amount'].sum():,.2f}")
        print(f"✓ Average Transaction: ₹{df['amount'].mean():.2f}")
        print(f"✓ Categories: {df['category'].nunique()}")
        print(f"✓ Merchants: {df['merchant'].nunique()}")
        
        df['year_month'] = df['date'].dt.to_period('M')
        monthly_summary = df.groupby('year_month')['amount'].agg(['sum', 'count', 'mean'])
        
        print("\n📈 Monthly Spending Pattern:")
        print(monthly_summary.to_string())
        
        print("\n🎯 Category-wise Spending:")
        category_summary = df.groupby('category')['amount'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
        category_summary['percentage'] = (category_summary['sum'] / df['amount'].sum() * 100).round(2)
        print(category_summary.to_string())
        
        return df
    
    def engineer_features(self, df):
        """Feature engineering"""
        self.print_header("🔧 FEATURE ENGINEERING")
        
        df = df.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['week_of_month'] = ((df['day'] - 1) // 7 + 1)
        df['quarter'] = df['date'].dt.quarter
        df['is_month_start'] = (df['day'] <= 7).astype(int)
        df['is_month_end'] = (df['day'] >= 24).astype(int)
        
        print("✓ Time-based features created:")
        print("  - Year, Month, Day, Day of week")
        print("  - Weekend indicator, Quarter")
        print("  - Month start/end indicators")
        
        return df
    
    def create_monthly_features(self, df):
        """Create monthly aggregated features"""
        self.print_header("📊 MONTHLY FEATURE AGGREGATION")
        
        df['year_month'] = df['date'].dt.to_period('M')
        
        monthly = df.groupby('year_month').agg({
            'amount': ['sum', 'mean', 'std', 'min', 'max', 'count'],
            'is_recurring': 'sum',
            'is_weekend': 'mean'
        }).reset_index()
        
        monthly.columns = ['year_month', 'total_amount', 'avg_amount', 'std_amount', 
                          'min_amount', 'max_amount', 'num_transactions', 
                          'recurring_count', 'weekend_ratio']
        
        category_pivot = df.groupby(['year_month', 'category'])['amount'].sum().unstack(fill_value=0)
        monthly = monthly.merge(category_pivot, left_on='year_month', right_index=True, how='left')
        
        print("\n✓ Creating lag features:")
        for lag in [1, 2, 3, 6]:
            monthly[f'total_lag_{lag}m'] = monthly['total_amount'].shift(lag)
            monthly[f'avg_lag_{lag}m'] = monthly['avg_amount'].shift(lag)
            monthly[f'count_lag_{lag}m'] = monthly['num_transactions'].shift(lag)
            print(f"  - Lag {lag} months")
        
        print("\n✓ Creating rolling features:")
        for window in [2, 3, 6]:
            monthly[f'rolling_mean_{window}m'] = monthly['total_amount'].rolling(window=window).mean()
            monthly[f'rolling_std_{window}m'] = monthly['total_amount'].rolling(window=window).std()
            print(f"  - {window} month rolling")
        
        monthly['month_over_month_growth'] = monthly['total_amount'].pct_change()
        monthly = monthly.fillna(0)
        
        print(f"\n✓ Total features: {len(monthly.columns) - 1}")
        print(f"✓ Training samples: {len(monthly)}")
        
        return monthly

    def _print_metrics(self, model_name, y_true, y_pred):
        """Print model metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        print(f"  RMSE:  ₹{rmse:,.2f}")
        print(f"  MAE:   ₹{mae:,.2f}")
        print(f"  R²:    {r2:.4f} ({r2*100:.2f}% accuracy)")
        print(f"  MAPE:  {mape:.2f}%")
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

    def _print_overfitting_check(self, y_train, pred_train, y_test, pred_test):
        """Check for overfitting"""
        r2_train = r2_score(y_train, pred_train)
        r2_test = r2_score(y_test, pred_test)
        gap = (r2_train - r2_test) * 100
        
        print(f"\n🔍 Overfitting Check:")
        print(f"  Train R²: {r2_train:.4f} ({r2_train*100:.2f}%)")
        print(f"  Test R²:  {r2_test:.4f} ({r2_test*100:.2f}%)")
        print(f"  Gap:      {gap:.2f}%", end="")
        
        if gap < 2:
            print(" ✅ Excellent - No overfitting")
        elif gap < 5:
            print(" ✅ Good - Minimal overfitting")
        elif gap < 10:
            print(" ⚠️  Moderate overfitting")
        else:
            print(" ❌ High overfitting - Model may not generalize well")
        
        return gap

    # ============================================================================
    # NEW: HYPERPARAMETER TUNING METHOD
    # ============================================================================
    def hyperparameter_tuning(self, X_train, y_train, cv_folds=3):
        """
        Perform hyperparameter tuning for all models using GridSearchCV/RandomizedSearchCV
        Returns tuned models and best parameters
        """
        self.print_header("🔬 HYPERPARAMETER TUNING")
        
        tuned_models = {}
        best_params = {}
        tuning_times = {}
        
        import time
        
        # ─────────────────────────────────────────────────────────────────────────
        # 1. Ridge Regression - GridSearchCV (small search space)
        # ─────────────────────────────────────────────────────────────────────────
        print("🔷 Tuning Ridge Regression...")
        print("  Parameter grid: alpha = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]")
        
        ridge_param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        }
        
        start_time = time.time()
        ridge_grid = GridSearchCV(
            Ridge(random_state=42),
            param_grid=ridge_param_grid,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        ridge_grid.fit(X_train, y_train)
        tuning_times['Ridge'] = time.time() - start_time
        
        tuned_models['ridge'] = ridge_grid.best_estimator_
        best_params['ridge'] = ridge_grid.best_params_
        
        print(f"  ✓ Best Parameters: {ridge_grid.best_params_}")
        print(f"  ✓ Best CV Score (neg_MSE): {ridge_grid.best_score_:,.2f}")
        print(f"  ✓ Time: {tuning_times['Ridge']:.2f}s")
        
        # ─────────────────────────────────────────────────────────────────────────
        # 2. Random Forest - RandomizedSearchCV (larger search space)
        # ─────────────────────────────────────────────────────────────────────────
        print("\n🔷 Tuning Random Forest...")
        print("  Parameter distributions:")
        print("    - n_estimators: [50, 100, 150, 200, 250]")
        print("    - max_depth: [3, 5, 7, 10, 15, 20, None]")
        print("    - min_samples_split: [2, 5, 10]")
        print("    - min_samples_leaf: [1, 2, 4]")
        
        rf_param_dist = {
            'n_estimators': [50, 100, 150, 200, 250],
            'max_depth': [3, 5, 7, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        start_time = time.time()
        rf_random = RandomizedSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_distributions=rf_param_dist,
            n_iter=30,  # Try 30 random combinations
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        rf_random.fit(X_train, y_train)
        tuning_times['Random Forest'] = time.time() - start_time
        
        tuned_models['random_forest'] = rf_random.best_estimator_
        best_params['random_forest'] = rf_random.best_params_
        
        print(f"  ✓ Best Parameters: {rf_random.best_params_}")
        print(f"  ✓ Best CV Score (neg_MSE): {rf_random.best_score_:,.2f}")
        print(f"  ✓ Time: {tuning_times['Random Forest']:.2f}s")
        
        # ─────────────────────────────────────────────────────────────────────────
        # 3. Gradient Boosting - RandomizedSearchCV
        # ─────────────────────────────────────────────────────────────────────────
        print("\n🔷 Tuning Gradient Boosting...")
        print("  Parameter distributions:")
        print("    - n_estimators: [50, 100, 150, 200]")
        print("    - learning_rate: [0.01, 0.05, 0.1, 0.2]")
        print("    - max_depth: [3, 4, 5, 6, 7]")
        
        gb_param_dist = {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        start_time = time.time()
        gb_random = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_distributions=gb_param_dist,
            n_iter=30,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        gb_random.fit(X_train, y_train)
        tuning_times['Gradient Boosting'] = time.time() - start_time
        
        tuned_models['gradient_boosting'] = gb_random.best_estimator_
        best_params['gradient_boosting'] = gb_random.best_params_
        
        print(f"  ✓ Best Parameters: {gb_random.best_params_}")
        print(f"  ✓ Best CV Score (neg_MSE): {gb_random.best_score_:,.2f}")
        print(f"  ✓ Time: {tuning_times['Gradient Boosting']:.2f}s")
        
        # ─────────────────────────────────────────────────────────────────────────
        # 4. XGBoost - RandomizedSearchCV
        # ─────────────────────────────────────────────────────────────────────────
        print("\n🔷 Tuning XGBoost...")
        print("  Parameter distributions:")
        print("    - n_estimators: [50, 100, 150, 200]")
        print("    - learning_rate: [0.01, 0.05, 0.1, 0.2, 0.3]")
        print("    - max_depth: [3, 4, 5, 6, 7, 8]")
        
        xgb_param_dist = {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
        
        start_time = time.time()
        xgb_random = RandomizedSearchCV(
            xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
            param_distributions=xgb_param_dist,
            n_iter=30,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        xgb_random.fit(X_train, y_train)
        tuning_times['XGBoost'] = time.time() - start_time
        
        tuned_models['xgboost'] = xgb_random.best_estimator_
        best_params['xgboost'] = xgb_random.best_params_
        
        print(f"  ✓ Best Parameters: {xgb_random.best_params_}")
        print(f"  ✓ Best CV Score (neg_MSE): {xgb_random.best_score_:,.2f}")
        print(f"  ✓ Time: {tuning_times['XGBoost']:.2f}s")
        
        # Summary
        print("\n" + "─" * 80)
        print("📊 HYPERPARAMETER TUNING SUMMARY")
        print("─" * 80)
        total_time = sum(tuning_times.values())
        print(f"  Total tuning time: {total_time:.2f}s")
        for model_name, t in tuning_times.items():
            print(f"    {model_name}: {t:.2f}s")
        
        self.tuned_params = best_params
        return tuned_models, best_params

    def train_models(self, monthly_df, use_hyperparameter_tuning=True):
        """Train multiple ML models with optional hyperparameter tuning"""
        self.print_header("🤖 MODEL TRAINING")
        
        monthly_df = monthly_df.iloc[6:]
        
        if len(monthly_df) < 5:
            print("❌ ERROR: Need at least 12 months of data")
            return None
        
        exclude_cols = ['year_month', 'total_amount']
        feature_cols = [col for col in monthly_df.columns if col not in exclude_cols]
        
        X = monthly_df[feature_cols].values
        y = monthly_df['total_amount'].values
        
        self.feature_columns = feature_cols
        
        print(f"✓ Feature Matrix: {X.shape}")
        print(f"✓ Target Vector: {y.shape}")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler
        
        test_size = max(1, int(len(X) * 0.2))
        X_train, X_test = X_scaled[:-test_size], X_scaled[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        print(f"\n✓ Train Set: {len(X_train)} months")
        print(f"✓ Test Set: {len(X_test)} months")
        
        # ======================================================================
        # PHASE 1: Train with DEFAULT (non-tuned) parameters
        # ======================================================================
        self.print_header("📊 PHASE 1: TRAINING WITH DEFAULT PARAMETERS")
        
        default_results = {}
        predictions_default_train = {}
        predictions_default_test = {}
        
        # Linear Regression (no hyperparameters to tune)
        print("\n" + "─" * 80)
        print("🔷 Training Linear Regression (Default)...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred_train = lr.predict(X_train)
        lr_pred_test = lr.predict(X_test)
        predictions_default_train['Linear Regression'] = lr_pred_train
        predictions_default_test['Linear Regression'] = lr_pred_test
        self.models['linear'] = lr
        
        print("\n📊 Test Performance:")
        lr_metrics = self._print_metrics("Linear Regression", y_test, lr_pred_test)
        default_results['Linear Regression'] = lr_metrics
        
        # Ridge with DEFAULT alpha=1.0
        print("\n" + "─" * 80)
        print("🔷 Training Ridge Regression (Default: alpha=1.0)...")
        ridge_default = Ridge(alpha=1.0, random_state=42)
        ridge_default.fit(X_train, y_train)
        ridge_pred_train = ridge_default.predict(X_train)
        ridge_pred_test = ridge_default.predict(X_test)
        predictions_default_train['Ridge'] = ridge_pred_train
        predictions_default_test['Ridge'] = ridge_pred_test
        
        print("\n📊 Test Performance:")
        ridge_metrics = self._print_metrics("Ridge", y_test, ridge_pred_test)
        default_results['Ridge'] = ridge_metrics
        
        # Random Forest with DEFAULT parameters
        print("\n" + "─" * 80)
        print("🔷 Training Random Forest (Default: n_estimators=100, max_depth=10)...")
        rf_default = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_default.fit(X_train, y_train)
        rf_pred_train = rf_default.predict(X_train)
        rf_pred_test = rf_default.predict(X_test)
        predictions_default_train['Random Forest'] = rf_pred_train
        predictions_default_test['Random Forest'] = rf_pred_test
        
        print("\n📊 Test Performance:")
        rf_metrics = self._print_metrics("Random Forest", y_test, rf_pred_test)
        default_results['Random Forest'] = rf_metrics
        
        # Gradient Boosting with DEFAULT parameters
        print("\n" + "─" * 80)
        print("🔷 Training Gradient Boosting (Default: n_estimators=100, lr=0.1, max_depth=5)...")
        gb_default = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        gb_default.fit(X_train, y_train)
        gb_pred_train = gb_default.predict(X_train)
        gb_pred_test = gb_default.predict(X_test)
        predictions_default_train['Gradient Boosting'] = gb_pred_train
        predictions_default_test['Gradient Boosting'] = gb_pred_test
        
        print("\n📊 Test Performance:")
        gb_metrics = self._print_metrics("Gradient Boosting", y_test, gb_pred_test)
        default_results['Gradient Boosting'] = gb_metrics
        
        # XGBoost with DEFAULT parameters
        print("\n" + "─" * 80)
        print("🔷 Training XGBoost (Default: n_estimators=100, lr=0.1, max_depth=6)...")
        xgb_default = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
        xgb_default.fit(X_train, y_train)
        xgb_pred_train = xgb_default.predict(X_train)
        xgb_pred_test = xgb_default.predict(X_test)
        predictions_default_train['XGBoost'] = xgb_pred_train
        predictions_default_test['XGBoost'] = xgb_pred_test
        
        print("\n📊 Test Performance:")
        xgb_metrics = self._print_metrics("XGBoost", y_test, xgb_pred_test)
        default_results['XGBoost'] = xgb_metrics
        
        # Default Ensemble
        ensemble_default_train = (
            lr_pred_train * 0.1 + ridge_pred_train * 0.1 + 
            rf_pred_train * 0.25 + gb_pred_train * 0.25 + xgb_pred_train * 0.3
        )
        ensemble_default_test = (
            lr_pred_test * 0.1 + ridge_pred_test * 0.1 + 
            rf_pred_test * 0.25 + gb_pred_test * 0.25 + xgb_pred_test * 0.3
        )
        predictions_default_train['Ensemble'] = ensemble_default_train
        predictions_default_test['Ensemble'] = ensemble_default_test
        
        print("\n" + "─" * 80)
        print("🔷 Ensemble (Default)...")
        print("\n📊 Test Performance:")
        ensemble_metrics = self._print_metrics("Ensemble", y_test, ensemble_default_test)
        default_results['Ensemble'] = ensemble_metrics
        
        # ======================================================================
        # PHASE 2: HYPERPARAMETER TUNING (if enabled)
        # ======================================================================
        tuned_results = {}
        predictions_tuned_train = {}
        predictions_tuned_test = {}
        
        if use_hyperparameter_tuning:
            # Perform hyperparameter tuning
            tuned_models, best_params = self.hyperparameter_tuning(X_train, y_train, cv_folds=3)
            
            self.print_header("📊 PHASE 2: TRAINING WITH TUNED PARAMETERS")
            
            # Linear Regression (same as before - no tuning needed)
            predictions_tuned_train['Linear Regression'] = lr_pred_train
            predictions_tuned_test['Linear Regression'] = lr_pred_test
            tuned_results['Linear Regression'] = lr_metrics
            
            # Ridge with TUNED parameters
            print("\n" + "─" * 80)
            print(f"🔷 Ridge Regression (Tuned: {best_params['ridge']})...")
            ridge_tuned = tuned_models['ridge']
            ridge_tuned_pred_train = ridge_tuned.predict(X_train)
            ridge_tuned_pred_test = ridge_tuned.predict(X_test)
            predictions_tuned_train['Ridge'] = ridge_tuned_pred_train
            predictions_tuned_test['Ridge'] = ridge_tuned_pred_test
            self.models['ridge'] = ridge_tuned
            
            print("\n📊 Test Performance:")
            ridge_tuned_metrics = self._print_metrics("Ridge", y_test, ridge_tuned_pred_test)
            tuned_results['Ridge'] = ridge_tuned_metrics
            
            # Random Forest with TUNED parameters
            print("\n" + "─" * 80)
            print(f"🔷 Random Forest (Tuned: {best_params['random_forest']})...")
            rf_tuned = tuned_models['random_forest']
            rf_tuned_pred_train = rf_tuned.predict(X_train)
            rf_tuned_pred_test = rf_tuned.predict(X_test)
            predictions_tuned_train['Random Forest'] = rf_tuned_pred_train
            predictions_tuned_test['Random Forest'] = rf_tuned_pred_test
            self.models['random_forest'] = rf_tuned
            
            print("\n📊 Test Performance:")
            rf_tuned_metrics = self._print_metrics("Random Forest", y_test, rf_tuned_pred_test)
            tuned_results['Random Forest'] = rf_tuned_metrics
            
            # Feature importance from tuned RF
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_tuned.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            print("\n🌟 Top 10 Important Features (Tuned RF):")
            for idx, row in feature_importance.iterrows():
                print(f"  {row['feature']:30s} {row['importance']:.4f}")
            
            # Gradient Boosting with TUNED parameters
            print("\n" + "─" * 80)
            print(f"🔷 Gradient Boosting (Tuned: {best_params['gradient_boosting']})...")
            gb_tuned = tuned_models['gradient_boosting']
            gb_tuned_pred_train = gb_tuned.predict(X_train)
            gb_tuned_pred_test = gb_tuned.predict(X_test)
            predictions_tuned_train['Gradient Boosting'] = gb_tuned_pred_train
            predictions_tuned_test['Gradient Boosting'] = gb_tuned_pred_test
            self.models['gradient_boosting'] = gb_tuned
            
            print("\n📊 Test Performance:")
            gb_tuned_metrics = self._print_metrics("Gradient Boosting", y_test, gb_tuned_pred_test)
            tuned_results['Gradient Boosting'] = gb_tuned_metrics
            
            # XGBoost with TUNED parameters
            print("\n" + "─" * 80)
            print(f"🔷 XGBoost (Tuned: {best_params['xgboost']})...")
            xgb_tuned = tuned_models['xgboost']
            xgb_tuned_pred_train = xgb_tuned.predict(X_train)
            xgb_tuned_pred_test = xgb_tuned.predict(X_test)
            predictions_tuned_train['XGBoost'] = xgb_tuned_pred_train
            predictions_tuned_test['XGBoost'] = xgb_tuned_pred_test
            self.models['xgboost'] = xgb_tuned
            
            print("\n📊 Test Performance:")
            xgb_tuned_metrics = self._print_metrics("XGBoost", y_test, xgb_tuned_pred_test)
            tuned_results['XGBoost'] = xgb_tuned_metrics
            
            # Tuned Ensemble
            ensemble_tuned_train = (
                lr_pred_train * 0.1 + ridge_tuned_pred_train * 0.1 + 
                rf_tuned_pred_train * 0.25 + gb_tuned_pred_train * 0.25 + xgb_tuned_pred_train * 0.3
            )
            ensemble_tuned_test = (
                lr_pred_test * 0.1 + ridge_tuned_pred_test * 0.1 + 
                rf_tuned_pred_test * 0.25 + gb_tuned_pred_test * 0.25 + xgb_tuned_pred_test * 0.3
            )
            predictions_tuned_train['Ensemble'] = ensemble_tuned_train
            predictions_tuned_test['Ensemble'] = ensemble_tuned_test
            
            print("\n" + "─" * 80)
            print("🔷 Ensemble (Tuned)...")
            print("\n📊 Test Performance:")
            ensemble_tuned_metrics = self._print_metrics("Ensemble", y_test, ensemble_tuned_test)
            tuned_results['Ensemble'] = ensemble_tuned_metrics
            
            # ======================================================================
            # PHASE 3: BEFORE vs AFTER COMPARISON
            # ======================================================================
            self.print_header("📊 BEFORE vs AFTER HYPERPARAMETER TUNING COMPARISON")
            
            print("┌" + "─" * 98 + "┐")
            print(f"│ {'Model':<20} │ {'Metric':<8} │ {'Before (Default)':<18} │ {'After (Tuned)':<18} │ {'Improvement':<15} │")
            print("├" + "─" * 98 + "┤")
            
            for model_name in ['Ridge', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'Ensemble']:
                before = default_results[model_name]
                after = tuned_results[model_name]
                
                # RMSE comparison (lower is better)
                rmse_before = before['rmse']
                rmse_after = after['rmse']
                rmse_improvement = ((rmse_before - rmse_after) / rmse_before) * 100
                rmse_symbol = "✅" if rmse_improvement > 0 else "❌"
                
                print(f"│ {model_name:<20} │ {'RMSE':<8} │ ₹{rmse_before:>15,.2f} │ ₹{rmse_after:>15,.2f} │ {rmse_improvement:>+10.2f}% {rmse_symbol} │")
                
                # R² comparison (higher is better)
                r2_before = before['r2']
                r2_after = after['r2']
                r2_improvement = (r2_after - r2_before) * 100
                r2_symbol = "✅" if r2_improvement > 0 else "❌"
                
                print(f"│ {'':<20} │ {'R²':<8} │ {r2_before:>17.4f} │ {r2_after:>17.4f} │ {r2_improvement:>+10.2f}% {r2_symbol} │")
                
                # MAPE comparison (lower is better)
                mape_before = before['mape']
                mape_after = after['mape']
                mape_improvement = ((mape_before - mape_after) / mape_before) * 100 if mape_before != 0 else 0
                mape_symbol = "✅" if mape_improvement > 0 else "❌"
                
                print(f"│ {'':<20} │ {'MAPE':<8} │ {mape_before:>16.2f}% │ {mape_after:>16.2f}% │ {mape_improvement:>+10.2f}% {mape_symbol} │")
                print("├" + "─" * 98 + "┤")
            
            print("└" + "─" * 98 + "┘")
            
            # Summary statistics
            print("\n" + "=" * 80)
            print("📈 TUNING IMPACT SUMMARY")
            print("=" * 80)
            
            total_rmse_improvement = 0
            total_r2_improvement = 0
            improved_models = 0
            
            for model_name in ['Ridge', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'Ensemble']:
                before = default_results[model_name]
                after = tuned_results[model_name]
                
                rmse_imp = ((before['rmse'] - after['rmse']) / before['rmse']) * 100
                r2_imp = (after['r2'] - before['r2']) * 100
                
                total_rmse_improvement += rmse_imp
                total_r2_improvement += r2_imp
                
                if rmse_imp > 0:
                    improved_models += 1
            
            avg_rmse_improvement = total_rmse_improvement / 5
            avg_r2_improvement = total_r2_improvement / 5
            
            print(f"\n✓ Models improved: {improved_models}/5")
            print(f"✓ Average RMSE improvement: {avg_rmse_improvement:+.2f}%")
            print(f"✓ Average R² improvement: {avg_r2_improvement:+.2f}%")
            
            # Best model selection
            best_model_name = min(tuned_results.keys(), key=lambda x: tuned_results[x]['rmse'])
            best_model_metrics = tuned_results[best_model_name]
            
            print(f"\n🏆 BEST MODEL AFTER TUNING: {best_model_name}")
            print(f"   RMSE: ₹{best_model_metrics['rmse']:,.2f}")
            print(f"   R²: {best_model_metrics['r2']:.4f}")
            print(f"   MAPE: {best_model_metrics['mape']:.2f}%")
            
            # Store for later use
            self.tuning_results = {
                'default': default_results,
                'tuned': tuned_results,
                'best_params': best_params
            }
            
            # Use tuned predictions for return
            final_predictions = predictions_tuned_test
            
        else:
            # No tuning - use default models
            self.models['ridge'] = ridge_default
            self.models['random_forest'] = rf_default
            self.models['gradient_boosting'] = gb_default
            self.models['xgboost'] = xgb_default
            final_predictions = predictions_default_test
        
        # Store training history
        self.training_history = {
            'X_test': X_test,
            'y_test': y_test,
            'predictions': final_predictions,
            'monthly_df': monthly_df
        }
        
        return final_predictions

    def predict_next_month(self, monthly_df):
        """Predict next month"""
        self.print_header("🔮 NEXT MONTH PREDICTION")
        
        last_row = monthly_df.iloc[-1]
        X_pred = last_row[self.feature_columns].values.reshape(1, -1)
        X_pred_scaled = self.scalers['main'].transform(X_pred)
        
        print("📊 Individual Model Predictions:")
        
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(X_pred_scaled)[0]
            predictions[name] = pred
            print(f"  {name:20s}: ₹{pred:,.2f}")
        
        ensemble_pred = (
            predictions['linear'] * 0.1 +
            predictions['ridge'] * 0.1 +
            predictions['random_forest'] * 0.25 +
            predictions['gradient_boosting'] * 0.25 +
            predictions['xgboost'] * 0.3
        )
        
        print(f"  {'ENSEMBLE':20s}: ₹{ensemble_pred:,.2f}")
        
        pred_values = list(predictions.values())
        pred_std = np.std(pred_values)
        pred_mean = np.mean(pred_values)
        confidence = max(0, min(100, 100 - (pred_std / pred_mean) * 100))
        
        lower_bound = ensemble_pred - pred_std
        upper_bound = ensemble_pred + pred_std
        
        print(f"\n📈 Prediction Analysis:")
        print(f"  Confidence Score: {confidence:.1f}%")
        print(f"  Prediction Range: ₹{lower_bound:,.2f} - ₹{upper_bound:,.2f}")
        print(f"  Standard Deviation: ₹{pred_std:,.2f}")
        
        hist_avg = monthly_df['total_amount'].mean()
        diff_from_avg = ((ensemble_pred - hist_avg) / hist_avg) * 100
        
        print(f"\n📊 Historical Comparison:")
        print(f"  Historical Average: ₹{hist_avg:,.2f}")
        print(f"  Predicted vs Average: {diff_from_avg:+.1f}%")
        
        if diff_from_avg > 10:
            print(f"  ⚠️  WARNING: {diff_from_avg:.1f}% higher than usual!")
        elif diff_from_avg < -10:
            print(f"  ✅ GOOD: {abs(diff_from_avg):.1f}% lower!")
        else:
            print(f"  ✅ NORMAL: Spending in expected range")
        
        return {
            'predicted_amount': ensemble_pred,
            'confidence': confidence,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'individual_predictions': predictions
        }
    
    def train_category_models(self, df):
        """Train category models"""
        self.print_header("🎯 CATEGORY-WISE PREDICTION")
        
        df['year_month'] = df['date'].dt.to_period('M')
        category_predictions = {}
        
        for category in sorted(df['category'].unique()):
            cat_df = df[df['category'] == category]
            monthly_cat = cat_df.groupby('year_month')['amount'].sum().reset_index()
            monthly_cat.columns = ['year_month', 'amount']
            
            if len(monthly_cat) < 4:
                continue
            
            for lag in [1, 2, 3]:
                monthly_cat[f'lag_{lag}'] = monthly_cat['amount'].shift(lag)
            
            monthly_cat = monthly_cat.dropna()
            
            if len(monthly_cat) < 3:
                continue
            
            X = monthly_cat[[f'lag_{i}' for i in [1, 2, 3]]].values
            y = monthly_cat['amount'].values
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            last_features = X[-1].reshape(1, -1)
            prediction = model.predict(last_features)[0]
            
            self.category_models[category] = model
            category_predictions[category] = max(0, prediction)
            
            hist_avg = monthly_cat['amount'].mean()
            
            print(f"\n  {category}:")
            print(f"    Predicted: ₹{prediction:,.2f}")
            print(f"    Historical Avg: ₹{hist_avg:,.2f}")
            print(f"    Change: {((prediction - hist_avg) / hist_avg * 100):+.1f}%")
        
        print("\n" + "─" * 80)
        print(f"  💰 Total Predicted: ₹{sum(category_predictions.values()):,.2f}")
        
        return category_predictions
    
    def generate_insights(self, df, prediction_result, category_predictions):
        """Generate insights"""
        self.print_header("💡 ACTIONABLE INSIGHTS")
        
        df['year_month'] = df['date'].dt.to_period('M')
        last_month = df['year_month'].max()
        last_month_spending = df[df['year_month'] == last_month]['amount'].sum()
        
        predicted = prediction_result['predicted_amount']
        diff = predicted - last_month_spending
        pct_diff = (diff / last_month_spending) * 100
        
        print(f"📈 SPENDING TREND:")
        print(f"  Last Month: ₹{last_month_spending:,.2f}")
        print(f"  Next Month Predicted: ₹{predicted:,.2f}")
        print(f"  Change: ₹{diff:,.2f} ({pct_diff:+.1f}%)")
        
        if pct_diff > 10:
            print(f"  ⚠️  HIGH ALERT: Significant increase!")
        elif pct_diff < -10:
            print(f"  ✅ GOOD: Spending decreasing!")
        
        print(f"\n🎯 TOP CATEGORIES:")
        last_month_df = df[df['year_month'] == last_month]
        top_categories = last_month_df.groupby('category')['amount'].sum().sort_values(ascending=False).head(5)
        
        for idx, (cat, amount) in enumerate(top_categories.items(), 1):
            pct = (amount / last_month_spending) * 100
            print(f"  {idx}. {cat:20s} ₹{amount:,.2f} ({pct:.1f}%)")
        
        print(f"\n💰 SAVINGS OPPORTUNITIES:")
        
        for category, pred_amount in category_predictions.items():
            cat_df = df[df['category'] == category]
            recent_avg = cat_df[cat_df['year_month'] == last_month]['amount'].sum()
            
            if recent_avg > 0:
                increase = ((pred_amount - recent_avg) / recent_avg) * 100
                
                if increase > 15:
                    savings_potential = (pred_amount - recent_avg) * 0.3
                    print(f"  • {category}: {increase:.0f}% increase")
                    print(f"    Potential savings: ₹{savings_potential:,.2f}")
        
        print(f"\n✅ RECOMMENDATIONS:")
        print(f"  • Set budget alert at ₹{predicted * 0.9:,.2f}")
        print(f"  • Review unused subscriptions")
        print(f"  • Plan major purchases carefully")
        print(f"  • Track daily spending")
    
    def save_predictions_to_csv(self, df, monthly_df, prediction_result, category_predictions):
        """Save predictions to CSV for dashboard"""
        self.print_header("💾 SAVING PREDICTIONS")
        
        # Create results directory if it doesn't exist
        os.makedirs('../results', exist_ok=True)
        
        # Get last date from data
        last_date = pd.to_datetime(df['date']).max()
        
        # Create predictions for next 6 months
        predictions_list = []
        for i in range(1, 7):
            pred_date = last_date + timedelta(days=30*i)
            pred_month = pred_date.strftime('%Y-%m')
            
            # Use ensemble prediction with slight variation for each month
            base_pred = prediction_result['predicted_amount']
            variation = np.random.uniform(0.95, 1.05)
            pred_amount = base_pred * variation
            
            predictions_list.append({
                'prediction_month': pred_month,
                'prediction_date': pred_date.strftime('%Y-%m-%d'),
                'predicted_amount': round(pred_amount, 2),
                'confidence_score': round(prediction_result['confidence'] / 100, 4),
                'lower_bound': round(prediction_result['lower_bound'] * variation, 2),
                'upper_bound': round(prediction_result['upper_bound'] * variation, 2),
                'model': 'Ensemble (Tuned)' if self.tuned_params else 'Ensemble'
            })
        
        # Save main predictions
        pred_df = pd.DataFrame(predictions_list)
        pred_df.to_csv('../results/predictions.csv', index=False)
        print(f"✅ Saved predictions to: results/predictions.csv")
        print(f"   {len(pred_df)} months of predictions")
        
        # Save category predictions
        if category_predictions:
            cat_pred_list = []
            for category, amount in category_predictions.items():
                cat_pred_list.append({
                    'category': category,
                    'predicted_amount': round(amount, 2),
                    'prediction_month': (last_date + timedelta(days=30)).strftime('%Y-%m')
                })
            
            cat_pred_df = pd.DataFrame(cat_pred_list)
            cat_pred_df.to_csv('../results/category_predictions.csv', index=False)
            print(f"✅ Saved category predictions to: results/category_predictions.csv")
            print(f"   {len(cat_pred_df)} categories")
        
        # Save model performance metrics
        metrics_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_transactions': len(df),
            'date_range_start': df['date'].min().strftime('%Y-%m-%d'),
            'date_range_end': df['date'].max().strftime('%Y-%m-%d'),
            'next_month_prediction': round(prediction_result['predicted_amount'], 2),
            'confidence_score': round(prediction_result['confidence'], 2),
            'models_trained': len(self.models),
            'hyperparameter_tuning': 'Yes' if self.tuned_params else 'No'
        }
        
        # Add tuned parameters if available
        if self.tuned_params:
            for model_name, params in self.tuned_params.items():
                metrics_data[f'best_params_{model_name}'] = str(params)
        
        metrics_df = pd.DataFrame([metrics_data])
        metrics_df.to_csv('../results/prediction_metrics.csv', index=False)
        print(f"✅ Saved metrics to: results/prediction_metrics.csv")
        
        # NEW: Save tuning comparison results
        if self.tuning_results:
            comparison_data = []
            for model_name in self.tuning_results['default'].keys():
                before = self.tuning_results['default'][model_name]
                after = self.tuning_results['tuned'][model_name]
                
                comparison_data.append({
                    'model': model_name,
                    'rmse_before': round(before['rmse'], 2),
                    'rmse_after': round(after['rmse'], 2),
                    'rmse_improvement_%': round(((before['rmse'] - after['rmse']) / before['rmse']) * 100, 2),
                    'r2_before': round(before['r2'], 4),
                    'r2_after': round(after['r2'], 4),
                    'r2_improvement_%': round((after['r2'] - before['r2']) * 100, 2),
                    'mape_before': round(before['mape'], 2),
                    'mape_after': round(after['mape'], 2)
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv('../results/tuning_comparison.csv', index=False)
            print(f"✅ Saved tuning comparison to: results/tuning_comparison.csv")
        
        print(f"\n📊 Dashboard files ready!")


def main():
    print("\n" + "="*80)
    print("  🚀 OPAM: Next Month Expense Predictor with Hyperparameter Tuning")
    print("="*80)
    
    predictor = OPAMExpensePredictor()
    
    print("\n📂 Loading data...")
    df = pd.read_csv('../data/transactions.csv')
    
    df = predictor.load_and_prepare_data(df)
    df = predictor.engineer_features(df)
    monthly_df = predictor.create_monthly_features(df)
    
    # Train with hyperparameter tuning enabled (set to False to disable)
    predictor.train_models(monthly_df, use_hyperparameter_tuning=True)
    
    prediction_result = predictor.predict_next_month(monthly_df)
    category_predictions = predictor.train_category_models(df)
    predictor.generate_insights(df, prediction_result, category_predictions)
    
    # Save predictions to CSV
    predictor.save_predictions_to_csv(df, monthly_df, prediction_result, category_predictions)
    
    predictor.print_header("✅ ANALYSIS COMPLETE")
    print("\n🎉 All predictions saved! Dashboard is ready to use.")
    print("\nNext steps:")
    print("  1. Check results/predictions.csv")
    print("  2. Check results/tuning_comparison.csv for before/after comparison")
    print("  3. Run: streamlit run opam_dashboard_fixed.py")
    print("  4. View predictions on dashboard!")
    print()

if __name__ == "__main__":
    main()