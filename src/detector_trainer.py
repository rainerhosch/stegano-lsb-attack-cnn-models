import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import yaml
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class DetectorTrainer:
    def __init__(self, config_path="configs/detector_config.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config atau gunakan default
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        # Inisialisasi model dan storage
        self.models = {}
        self.results = {}
        self.scaler = None
        self.feature_importance = {}
        
    def _get_default_config(self):
        """Default configuration untuk training detektor"""
        return {
            'training': {
                'test_size': 0.3,  # Increased for small dataset
                'random_state': 42,
                'cv_folds': 3,     # Reduced for small dataset
                'stratified': True
            },
            'models': {
                'random_forest': {
                    'n_estimators': 50,    # Reduced for small dataset
                    'max_depth': 5,        # Reduced to prevent overfitting
                    'min_samples_split': 5, # Increased for small dataset
                    'min_samples_leaf': 3,  # Increased for small dataset
                    'random_state': 42
                },
                'xgboost': {
                    'n_estimators': 50,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                'lightgbm': {
                    'n_estimators': 50,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'min_child_samples': 5,  # Important for small dataset
                    'random_state': 42
                },
                'svm': {
                    'C': 0.1,      # Reduced for small dataset
                    'kernel': 'rbf',
                    'probability': True,
                    'random_state': 42
                },
                'logistic_regression': {
                    'C': 0.1,      # Reduced for regularization
                    'random_state': 42,
                    'max_iter': 1000,
                    'penalty': 'l2'
                },
                'gradient_boosting': {
                    'n_estimators': 50,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                'knn': {
                    'n_neighbors': 3,  # Reduced for small dataset
                    'weights': 'distance'
                }
            },
            'feature_selection': {
                'enabled': True,
                'method': 'importance',
                'top_k_features': 100,  # Naikkan dari 20 ke 100
                'variance_threshold': 0.01  # Remove low variance features
            },
            'evaluation': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                'save_plots': True,
                'cross_validate': True
            }
        }
    
    def load_features(self, features_path="data/processed/steganalysis_features_20251012_164824.csv"):
        """Load features dataset"""
        print("=== LOADING FEATURES DATASET ===")
        
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        # Load dataset
        self.features_df = pd.read_csv(features_path)
        print(f"Dataset loaded: {self.features_df.shape}")
        
        # Check for NaN values
        nan_count = self.features_df.isna().sum().sum()
        if nan_count > 0:
            print(f"Warning: Found {nan_count} NaN values. Filling with 0.")
            self.features_df = self.features_df.fillna(0)
        
        # Check label distribution
        label_counts = self.features_df['label'].value_counts()
        print(f"Label distribution:\n{label_counts}")
        
        # Check if we have enough samples
        if len(self.features_df) < 20:
            print("Warning: Very small dataset. Results may be unreliable.")
        
        # Prepare feature matrix and labels
        exclude_cols = ['model_path', 'model_name', 'model_type', 'label']
        if 'injection_payload_type' in self.features_df.columns:
            exclude_cols.extend(['injection_payload_type', 'injection_lsb_bits', 'injection_ratio'])
        
        self.feature_cols = [col for col in self.features_df.columns if col not in exclude_cols]
        
        X = self.features_df[self.feature_cols].values
        y = self.features_df['label'].values
        
        print(f"Feature matrix: {X.shape}")
        print(f"Number of features: {len(self.feature_cols)}")
        
        return X, y
    
    # def _remove_low_variance_features(self, X, feature_names):
    #     """Remove features with very low variance"""
    #     from sklearn.feature_selection import VarianceThreshold
        
    #     variance_threshold = self.config['feature_selection']['variance_threshold']
    #     selector = VarianceThreshold(threshold=variance_threshold)
    #     X_high_variance = selector.fit_transform(X)
        
    #     # Get selected feature indices
    #     selected_indices = selector.get_support(indices=True)
    #     selected_features = [feature_names[i] for i in selected_indices]
        
    #     print(f"Removed {X.shape[1] - X_high_variance.shape[1]} low-variance features")
    #     print(f"Remaining features: {X_high_variance.shape[1]}")
        
    #     return X_high_variance, selected_features
    
    def _remove_low_variance_features(self, X, feature_names):
        """Remove features with very low variance"""
        from sklearn.feature_selection import VarianceThreshold
        
        # Gunakan default value jika config tidak ada
        try:
            variance_threshold = self.config['feature_selection']['variance_threshold']
        except KeyError:
            variance_threshold = 0.01  # Default value
            print("Using default variance_threshold: 0.01")
        
        try:
            selector = VarianceThreshold(threshold=variance_threshold)
            X_high_variance = selector.fit_transform(X)
            
            # Get selected feature indices
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
            
            print(f"Removed {X.shape[1] - X_high_variance.shape[1]} low-variance features")
            print(f"Remaining features: {X_high_variance.shape[1]}")
            
            return X_high_variance, selected_features
            
        except Exception as e:
            print(f"Variance threshold failed: {e}. Using original features.")
            return X, feature_names

    # def preprocess_features(self, X, y):
    #     """Preprocess features: scaling dan feature selection"""
    #     print("\n=== PREPROCESSING FEATURES ===")
        
    #     # Handle missing values
    #     X = np.nan_to_num(X)
        
    #     # Remove low variance features first
    #     X_high_variance, high_variance_features = self._remove_low_variance_features(X, self.feature_cols)
    #     self.feature_cols = high_variance_features
    #     X = X_high_variance
        
    #     # Split data
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, 
    #         test_size=self.config['training']['test_size'],
    #         random_state=self.config['training']['random_state'],
    #         stratify=y if self.config['training']['stratified'] else None
    #     )
        
    #     print(f"Training set: {X_train.shape}, Labels: {np.bincount(y_train)}")
    #     print(f"Test set: {X_test.shape}, Labels: {np.bincount(y_test)}")
        
    #     # Feature scaling
    #     self.scaler = StandardScaler()  # Switch to StandardScaler for better performance
    #     X_train_scaled = self.scaler.fit_transform(X_train)
    #     X_test_scaled = self.scaler.transform(X_test)
        
    #     # Feature selection
    #     if self.config['feature_selection']['enabled'] and X_train_scaled.shape[1] > 10:
    #         X_train_processed, X_test_processed, selected_features = self._select_features(
    #             X_train_scaled, y_train, X_test_scaled
    #         )
    #         print(f"Selected {len(selected_features)} features")
    #     else:
    #         X_train_processed, X_test_processed = X_train_scaled, X_test_scaled
    #         selected_features = self.feature_cols
        
    #     self.selected_features = selected_features
        
    #     return X_train_processed, X_test_processed, y_train, y_test
    
    def preprocess_features(self, X, y):
        """Preprocess features: scaling dan feature selection"""
        print("\n=== PREPROCESSING FEATURES ===")
        
        # Handle missing values
        X = np.nan_to_num(X)
        
        # Remove low variance features first (dengan error handling)
        try:
            X_high_variance, high_variance_features = self._remove_low_variance_features(X, self.feature_cols)
            self.feature_cols = high_variance_features
            X = X_high_variance
        except Exception as e:
            print(f"Warning: Variance removal failed: {e}. Using all features.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state'],
            stratify=y if self.config['training']['stratified'] else None
        )
        
        print(f"Training set: {X_train.shape}, Labels: {np.bincount(y_train)}")
        print(f"Test set: {X_test.shape}, Labels: {np.bincount(y_test)}")
        
        # Feature scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection (dengan pengecekan jumlah features)
        if (self.config['feature_selection']['enabled'] and 
            X_train_scaled.shape[1] > 10 and 
            len(self.feature_cols) > 10):
            
            X_train_processed, X_test_processed, selected_features = self._select_features(
                X_train_scaled, y_train, X_test_scaled
            )
            print(f"Selected {len(selected_features)} features")
        else:
            X_train_processed, X_test_processed = X_train_scaled, X_test_scaled
            selected_features = self.feature_cols
            print(f"Using all {len(selected_features)} features (feature selection skipped)")
        
        self.selected_features = selected_features
        
        return X_train_processed, X_test_processed, y_train, y_test

    def _select_features(self, X_train, y_train, X_test):
        """Feature selection menggunakan importance dari Random Forest"""
        print("Performing feature selection...")
        
        # Train Random Forest untuk feature importance
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=self.config['training']['random_state']
        )
        rf.fit(X_train, y_train)
        
        # Get feature importance
        importance_scores = rf.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Select top k features
        top_k = min(self.config['feature_selection']['top_k_features'], len(self.feature_cols))
        selected_features = feature_importance_df.head(top_k)['feature'].tolist()
        selected_indices = [self.feature_cols.index(f) for f in selected_features]
        
        # Filter features
        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
        
        # Save feature importance
        self.feature_importance['rf'] = feature_importance_df
        
        print(f"Top 5 features: {selected_features[:5]}")
        
        return X_train_selected, X_test_selected, selected_features
    
    def initialize_models(self):
        """Initialize semua model yang akan di-training"""
        print("\n=== INITIALIZING MODELS ===")
        
        models_config = self.config['models']
        
        # Tree-based models
        self.models['random_forest'] = RandomForestClassifier(**models_config['random_forest'])
        self.models['xgboost'] = XGBClassifier(**models_config['xgboost'])
        

        # FIX: LightGBM dengan disabled parallel processing
        lgbm_config = models_config['lightgbm'].copy()
        lgbm_config.update({
            'n_jobs': 1,
            'force_row_wise': True,
            'min_child_samples': 20,    # Kurangi untuk small dataset
            'min_data_in_leaf': 20,     # Kurangi minimum requirements
            'feature_fraction': 0.8,   # Gunakan sebagian feature
        })
        self.models['lightgbm'] = LGBMClassifier(**lgbm_config)
        # self.models['lightgbm'] = LGBMClassifier(**models_config['lightgbm'])
        self.models['gradient_boosting'] = GradientBoostingClassifier(**models_config['gradient_boosting'])
        
        # Linear models
        self.models['svm'] = SVC(**models_config['svm'])
        self.models['logistic_regression'] = LogisticRegression(**models_config['logistic_regression'])
        
        # Distance-based
        self.models['knn'] = KNeighborsClassifier(**models_config['knn'])
        
        print(f"Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train semua model dan evaluate performance"""
        print("\n=== TRAINING MODELS ===")
        
        self.results = {}
        
        for model_name, model in tqdm(self.models.items(), desc="Training models"):
            print(f"\n--- Training {model_name} ---")
            
            try:
                # Training
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                
                # Evaluation
                model_results = self._evaluate_model(y_test, y_pred, y_pred_proba, model_name)
                
                # Cross-validation jika dienable
                if self.config['evaluation']['cross_validate'] and len(np.unique(y_train)) > 1:
                    cv_scores = self._cross_validate_model(model, X_train, y_train)
                    model_results['cv_scores'] = cv_scores
                    print(f"CV Accuracy: {cv_scores['mean_accuracy']:.4f} (+/- {cv_scores['std_accuracy']:.4f})")
                
                # Feature importance untuk tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = pd.DataFrame({
                        'feature': self.selected_features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                
                self.results[model_name] = model_results
                self.models[model_name] = model  # Update trained model
                
                print(f"✓ {model_name} completed")
                
            except Exception as e:
                print(f"✗ Error training {model_name}: {e}")
                continue
        
        return self.results
    
    def _evaluate_model(self, y_true, y_pred, y_pred_proba, model_name):
        """Evaluate model performance"""
        results = {}
        
        # Basic metrics dengan zero_division handling
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, zero_division=0)
        results['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC-AUC jika probabilities available
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            try:
                results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                results['roc_auc'] = 0.5  # Default for random classifier
        else:
            results['roc_auc'] = 0.5
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report dengan error handling
        try:
            report = classification_report(y_true, y_pred, output_dict=True)
            results['classification_report'] = report
        except:
            results['classification_report'] = {'warning': 'Could not generate classification report'}
        
        # Print results
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1']:.4f}")
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
        
        return results
    
    def _cross_validate_model(self, model, X, y):
        """Perform cross-validation"""
        try:
            cv = StratifiedKFold(
                n_splits=min(self.config['training']['cv_folds'], len(np.unique(y))),
                shuffle=True,
                random_state=self.config['training']['random_state']
            )
            
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            
            return {
                'mean_accuracy': np.mean(scores),
                'std_accuracy': np.std(scores),
                'all_scores': scores.tolist()
            }
        except Exception as e:
            print(f"Cross-validation failed: {e}")
            return {
                'mean_accuracy': 0.5,
                'std_accuracy': 0.0,
                'all_scores': []
            }
    
    def compare_models(self):
        """Bandingkan performance semua model"""
        print("\n=== MODEL COMPARISON ===")
        
        if not self.results:
            print("No results to compare. Train models first.")
            return
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            row = {
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1'],
                'ROC-AUC': results['roc_auc']
            }
            
            if 'cv_scores' in results:
                row['CV_Accuracy'] = results['cv_scores']['mean_accuracy']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by F1-Score, then by Accuracy
        comparison_df = comparison_df.sort_values(['F1-Score', 'Accuracy'], ascending=[False, False])
        
        print("\nModel Performance Comparison:")
        print(comparison_df.round(4))
        
        # Save comparison results
        os.makedirs('data/results', exist_ok=True)
        comparison_df.to_csv('data/results/model_comparison.csv', index=False)
        print(f"\nComparison saved to: data/results/model_comparison.csv")
        
        return comparison_df
    
    def plot_results(self):
        """Plot hasil evaluation"""
        if not self.results:
            print("No results to plot. Train models first.")
            return
        
        print("\n=== GENERATING PLOTS ===")
        
        os.makedirs('data/results/plots', exist_ok=True)
        
        try:
            # 1. Model comparison bar plot
            self._plot_model_comparison()
            
            # 2. Confusion matrices
            self._plot_confusion_matrices()
            
            # 3. Feature importance
            self._plot_feature_importance()
            
            print("All plots saved to: data/results/plots/")
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    def _plot_model_comparison(self):
        """Plot perbandingan model"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        model_names = list(self.results.keys())
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                scores = [self.results[model].get(metric.lower(), 0) for model in model_names]
                bars = axes[i].bar(model_names, scores, color='skyblue', alpha=0.7)
                axes[i].set_title(f'{metric} Comparison')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for j, v in enumerate(scores):
                    axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('data/results/plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices untuk semua model"""
        n_models = len(self.results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (model_name, results) in enumerate(self.results.items()):
            if i < len(axes):
                cm = results['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_title(f'Confusion Matrix - {model_name}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('data/results/plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self):
        """Plot feature importance untuk tree-based models"""
        tree_models = ['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting']
        available_models = [model for model in tree_models if model in self.feature_importance]
        
        if not available_models:
            print("No feature importance data available for tree-based models")
            return
        
        n_models = len(available_models)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for i, model_name in enumerate(available_models):
            importance_df = self.feature_importance[model_name].head(10)
            axes[i].barh(importance_df['feature'], importance_df['importance'])
            axes[i].set_title(f'Feature Importance - {model_name}')
            axes[i].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('data/results/plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_models(self):
        """Save trained models dan preprocessing objects"""
        print("\n=== SAVING MODELS ===")
        
        os.makedirs('models/detector', exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            if hasattr(model, 'fit'):  # Pastikan model sudah trained
                model_path = f'models/detector/{model_name}_detector.pkl'
                joblib.dump(model, model_path)
                print(f"Saved {model_name} to: {model_path}")
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = 'models/detector/feature_scaler.pkl'
            joblib.dump(self.scaler, scaler_path)
            print(f"Saved scaler to: {scaler_path}")
        
        # Save feature list
        feature_info = {
            'all_features': self.feature_cols,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance
        }
        
        with open('models/detector/feature_info.pkl', 'wb') as f:
            pickle.dump(feature_info, f)
        print(f"Saved feature info to: models/detector/feature_info.pkl")
        
        # Save training config
        with open('models/detector/training_config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        print(f"Saved training config to: models/detector/training_config.yaml")
    
    def train_detector(self):
        """Main training pipeline"""
        print("=== STEGANOGRAPHY DETECTOR TRAINING ===")
        
        try:
            # 1. Load features
            X, y = self.load_features()
            
            # 2. Preprocess features
            X_train, X_test, y_train, y_test = self.preprocess_features(X, y)
            
            # 3. Initialize models
            self.initialize_models()
            
            # 4. Train models
            self.train_models(X_train, X_test, y_train, y_test)
            
            # 5. Compare results
            comparison_df = self.compare_models()
            
            # 6. Generate plots
            if self.config['evaluation']['save_plots']:
                self.plot_results()
            
            # 7. Save models
            self.save_models()
            
            # 8. Print best model
            if not comparison_df.empty:
                best_model = comparison_df.iloc[0]
                print(f"\n🎉 BEST MODEL: {best_model['Model']}")
                print(f"   Accuracy: {best_model['Accuracy']:.4f}")
                print(f"   F1-Score: {best_model['F1-Score']:.4f}")
                print(f"   ROC-AUC: {best_model['ROC-AUC']:.4f}")
            else:
                print("\nNo successful models to compare")
            
            return self.results
            
        except Exception as e:
            print(f"Error in training pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None

def create_detector_config():
    """Create detector training configuration file"""
    config = {
        'training': {
            'test_size': 0.3,
            'random_state': 42,
            'cv_folds': 3,
            'stratified': True
        },
        'models': {
            'random_forest': {
                'n_estimators': 50,
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 3,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 50,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 50,
                'max_depth': 3,
                'learning_rate': 0.1,
                'min_child_samples': 5,
                'random_state': 42
            },
            'svm': {
                'C': 0.1,
                'kernel': 'rbf',
                'probability': True,
                'random_state': 42
            },
            'logistic_regression': {
                'C': 0.1,
                'random_state': 42,
                'max_iter': 1000,
                'penalty': 'l2'
            },
            'gradient_boosting': {
                'n_estimators': 50,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'knn': {
                'n_neighbors': 3,
                'weights': 'distance'
            }
        },
        'feature_selection': {
            'enabled': True,
            'method': 'importance',
            'top_k_features': 100,  # Naikkan dari 20 ke 100
            'variance_threshold': 0.01
        },
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            'save_plots': True,
            'cross_validate': True
        }
    }
    
    os.makedirs('configs', exist_ok=True)
    with open('configs/detector_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Detector config created: configs/detector_config.yaml")

if __name__ == "__main__":
    # Create config file jika belum ada
    if not os.path.exists("configs/detector_config.yaml"):
        create_detector_config()
    
    # Train detector
    trainer = DetectorTrainer()
    results = trainer.train_detector()
    
    if results:
        print("\n=== DETECTOR TRAINING COMPLETED ===")
        print("Models saved in: models/detector/")
        print("Results saved in: data/results/")
        
        # Print recommendations for improvement
        print("\n📈 RECOMMENDATIONS FOR IMPROVEMENT:")
        print("1. Generate more training data (aim for 100+ samples per class)")
        print("2. Experiment with different LSB injection parameters")
        print("3. Try neural network-based detectors")
        print("4. Add more diverse model architectures")
    else:
        print("\n=== TRAINING FAILED ===")