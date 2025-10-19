import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import yaml
import pickle
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    def __init__(self, config_path="configs/feature_config.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config atau gunakan default
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        # Inisialisasi storage untuk features
        self.features_df = None
        self.scaler = StandardScaler()
        
    def _get_default_config(self):
        """Default configuration untuk ekstraksi fitur"""
        return {
            'feature_extraction': {
                'weight_based_features': True,
                'activation_based_features': False,  # Butuh forward pass
                'statistical_moments': True,
                'lsb_analysis': True,
                'entropy_features': True,
                'batch_norm_features': True,
                'layer_wise_features': True,
                'correlation_features': True,
                'save_intermediate': True
            },
            'datasets': {
                'test_samples': 1000,  # Number of samples for activation analysis
                'batch_size': 64
            }
        }
    
    def _extract_weight_statistics(self, state_dict, layer_name, weights):
        """Ekstrak statistik dari bobot layer"""
        features = {}
        
        # Basic statistics
        weights_np = weights.cpu().numpy().flatten()
        
        features[f'{layer_name}_mean'] = np.mean(weights_np)
        features[f'{layer_name}_std'] = np.std(weights_np)
        features[f'{layer_name}_var'] = np.var(weights_np)
        features[f'{layer_name}_min'] = np.min(weights_np)
        features[f'{layer_name}_max'] = np.max(weights_np)
        features[f'{layer_name}_median'] = np.median(weights_np)
        
        # Statistical moments
        if len(weights_np) > 1:
            features[f'{layer_name}_skewness'] = stats.skew(weights_np)
            features[f'{layer_name}_kurtosis'] = stats.kurtosis(weights_np)
        else:
            features[f'{layer_name}_skewness'] = 0
            features[f'{layer_name}_kurtosis'] = 0
        
        # Range and percentiles
        features[f'{layer_name}_range'] = features[f'{layer_name}_max'] - features[f'{layer_name}_min']
        features[f'{layer_name}_q1'] = np.percentile(weights_np, 25)
        features[f'{layer_name}_q3'] = np.percentile(weights_np, 75)
        features[f'{layer_name}_iqr'] = features[f'{layer_name}_q3'] - features[f'{layer_name}_q1']
        
        return features
    
    def _extract_lsb_analysis(self, state_dict, layer_name, weights):
        """Analisis LSB pattern dari bobot"""
        features = {}
        
        try:
            weights_np = weights.cpu().numpy().flatten()
            
            # Convert to binary representation and analyze LSB
            lsb_bits = []
            for weight in weights_np[:1000]:  # Sample first 1000 weights for efficiency
                # Convert float32 to bytes to binary
                weight_bytes = weight.tobytes()
                binary_representation = ''.join(f'{byte:08b}' for byte in weight_bytes)
                
                # Take last few bits as LSB
                lsb = binary_representation[-4:]  # Last 4 bits
                lsb_bits.extend([int(bit) for bit in lsb])
            
            if lsb_bits:
                lsb_array = np.array(lsb_bits)
                
                # LSB statistics
                features[f'{layer_name}_lsb_mean'] = np.mean(lsb_array)
                features[f'{layer_name}_lsb_std'] = np.std(lsb_array)
                
                # LSB bias (deviation from 0.5)
                features[f'{layer_name}_lsb_bias'] = abs(np.mean(lsb_array) - 0.5)
                
                # LSB runs (consecutive same bits)
                runs = 1
                for i in range(1, len(lsb_array)):
                    if lsb_array[i] != lsb_array[i-1]:
                        runs += 1
                features[f'{layer_name}_lsb_runs'] = runs
                
                # LSB entropy
                unique, counts = np.unique(lsb_array, return_counts=True)
                prob = counts / len(lsb_array)
                features[f'{layer_name}_lsb_entropy'] = -np.sum(prob * np.log2(prob + 1e-10))
                
        except Exception as e:
            # Jika analisis LSB gagal, set default values
            features[f'{layer_name}_lsb_mean'] = 0.5
            features[f'{layer_name}_lsb_std'] = 0.5
            features[f'{layer_name}_lsb_bias'] = 0.0
            features[f'{layer_name}_lsb_runs'] = len(lsb_bits) if lsb_bits else 0
            features[f'{layer_name}_lsb_entropy'] = 1.0
        
        return features
    
    def _extract_entropy_features(self, state_dict, layer_name, weights):
        """Ekstrak fitur entropy dan complexity"""
        features = {}
        
        weights_np = weights.cpu().numpy().flatten()
        
        # Histogram entropy
        hist, _ = np.histogram(weights_np, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        features[f'{layer_name}_hist_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Signal complexity (approximate)
        diff_weights = np.diff(weights_np)
        features[f'{layer_name}_complexity'] = np.sum(diff_weights ** 2)
        
        return features
    
    def _extract_batch_norm_features(self, state_dict, layer_name, weights):
        """Fitur khusus untuk BatchNorm layers"""
        features = {}
        
        if 'running_mean' in layer_name or 'running_var' in layer_name:
            weights_np = weights.cpu().numpy().flatten()
            
            features[f'{layer_name}_bn_mean'] = np.mean(weights_np)
            features[f'{layer_name}_bn_std'] = np.std(weights_np)
            features[f'{layer_name}_bn_stability'] = 1.0 / (np.std(weights_np) + 1e-10)
        
        return features
    
    def _extract_correlation_features(self, state_dict):
        """Ekstrak fitur korelasi antar layer"""
        features = {}
        
        # Collect all weight tensors
        weight_tensors = []
        layer_names = []
        
        for layer_name, weights in state_dict.items():
            if 'weight' in layer_name and len(weights.shape) >= 2:
                weight_flat = weights.cpu().numpy().flatten()
                if len(weight_flat) > 100:  # Only use layers with sufficient parameters
                    weight_tensors.append(weight_flat[:1000])  # Use first 1000 for efficiency
                    layer_names.append(layer_name)
        
        # Calculate correlations between layers
        if len(weight_tensors) >= 2:
            corr_matrix = np.corrcoef(weight_tensors)
            
            # Extract correlation statistics
            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            if len(upper_tri) > 0:
                features['avg_layer_correlation'] = np.mean(upper_tri)
                features['max_layer_correlation'] = np.max(upper_tri)
                features['min_layer_correlation'] = np.min(upper_tri)
                features['std_layer_correlation'] = np.std(upper_tri)
        
        return features
    
    def _extract_global_features(self, state_dict):
        """Ekstrak fitur global dari seluruh model"""
        features = {}
        
        all_weights = []
        total_parameters = 0
        
        for layer_name, weights in state_dict.items():
            if 'weight' in layer_name:
                weights_flat = weights.cpu().numpy().flatten()
                all_weights.extend(weights_flat)
                total_parameters += weights.numel()
        
        if all_weights:
            all_weights = np.array(all_weights)
            
            # Global statistics
            features['global_mean'] = np.mean(all_weights)
            features['global_std'] = np.std(all_weights)
            features['global_skewness'] = stats.skew(all_weights)
            features['global_kurtosis'] = stats.kurtosis(all_weights)
            features['total_parameters'] = total_parameters
            
            # Global LSB analysis (sampled)
            sampled_weights = all_weights[:min(5000, len(all_weights))]
            lsb_pattern = []
            
            for weight in sampled_weights:
                weight_bytes = weight.tobytes()
                binary_rep = ''.join(f'{byte:08b}' for byte in weight_bytes)
                lsb_pattern.extend([int(bit) for bit in binary_rep[-2:]])  # Last 2 bits
            
            if lsb_pattern:
                lsb_array = np.array(lsb_pattern)
                features['global_lsb_bias'] = abs(np.mean(lsb_array) - 0.5)
                features['global_lsb_entropy'] = stats.entropy(
                    np.bincount(lsb_array, minlength=2) / len(lsb_array)
                )
        
        return features
    
    def extract_features_from_model(self, model_path, label=None, model_type=None):
        """Ekstrak semua fitur dari sebuah model"""
        print(f"Extracting features from: {os.path.basename(model_path)}")
        
        try:
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Extract metadata jika ada
                metadata = {}
                if 'injection_metadata' in checkpoint:
                    metadata = checkpoint['injection_metadata']
                    model_type = 'injected'
                elif label is None:
                    model_type = 'clean'
            else:
                state_dict = checkpoint
                model_type = model_type or 'clean'
            
            features = {}
            
            # Global features
            global_features = self._extract_global_features(state_dict)
            features.update(global_features)
            
            # Correlation features
            correlation_features = self._extract_correlation_features(state_dict)
            features.update(correlation_features)
            
            # Layer-wise features
            conv_layers = []
            fc_layers = []
            bn_layers = []
            
            for layer_name, weights in state_dict.items():
                # Kategorikan layer
                if 'conv' in layer_name and 'weight' in layer_name:
                    conv_layers.append((layer_name, weights))
                elif 'fc' in layer_name or 'classifier' in layer_name:
                    fc_layers.append((layer_name, weights))
                elif 'bn' in layer_name or 'batch_norm' in layer_name:
                    bn_layers.append((layer_name, weights))
                
                # Ekstrak fitur untuk setiap layer
                if self.config['feature_extraction']['weight_based_features']:
                    weight_features = self._extract_weight_statistics(state_dict, layer_name, weights)
                    features.update(weight_features)
                
                if self.config['feature_extraction']['lsb_analysis']:
                    lsb_features = self._extract_lsb_analysis(state_dict, layer_name, weights)
                    features.update(lsb_features)
                
                if self.config['feature_extraction']['entropy_features']:
                    entropy_features = self._extract_entropy_features(state_dict, layer_name, weights)
                    features.update(entropy_features)
                
                if self.config['feature_extraction']['batch_norm_features']:
                    bn_features = self._extract_batch_norm_features(state_dict, layer_name, weights)
                    features.update(bn_features)
            
            # Aggregate layer-specific features
            if conv_layers:
                conv_weights = torch.cat([w[1].flatten() for w in conv_layers])
                features['conv_weights_mean'] = conv_weights.mean().item()
                features['conv_weights_std'] = conv_weights.std().item()
            
            if fc_layers:
                fc_weights = torch.cat([w[1].flatten() for w in fc_layers])
                features['fc_weights_mean'] = fc_weights.mean().item()
                features['fc_weights_std'] = fc_weights.std().item()
            
            # Add metadata
            features['model_path'] = model_path
            features['model_name'] = os.path.basename(model_path)
            features['label'] = label if label is not None else (0 if model_type == 'clean' else 1)
            features['model_type'] = model_type
            
            # Add injection metadata jika ada
            if 'injection_metadata' in locals() and metadata:
                features['injection_payload_type'] = metadata.get('payload_type', 'none')
                features['injection_lsb_bits'] = metadata.get('num_lsb_bits', 0)
                features['injection_ratio'] = metadata.get('injection_ratio', 0.0)
            
            print(f"✓ Extracted {len(features)} features from {os.path.basename(model_path)}")
            return features
            
        except Exception as e:
            print(f"✗ Error extracting features from {model_path}: {e}")
            return None
    
    def extract_features_from_all_models(self, 
                                    clean_models_file="models/trained_models/models_list.txt",
                                    injected_models_file="models/stego_models_improved/models_list.txt"):
        """Ekstrak fitur dari semua model clean dan injected"""
        print("=== FEATURE EXTRACTION FROM ALL MODELS ===")
        
        all_features = []
        
        # Extract from clean models
        if os.path.exists(clean_models_file):
            print(f"\n--- Processing Clean Models ---")
            with open(clean_models_file, 'r') as f:
                clean_model_paths = [line.strip() for line in f if line.strip()]
            
            for model_path in tqdm(clean_model_paths, desc="Clean models"):
                if os.path.exists(model_path):
                    features = self.extract_features_from_model(model_path, label=0, model_type='clean')
                    if features:
                        all_features.append(features)
                else:
                    print(f"Model not found: {model_path}")
        
        # Extract from injected models
        if os.path.exists(injected_models_file):
            print(f"\n--- Processing Injected Models ---")
            with open(injected_models_file, 'r') as f:
                injected_model_paths = [line.strip() for line in f if line.strip()]
            
            for model_path in tqdm(injected_model_paths, desc="Injected models"):
                if os.path.exists(model_path):
                    features = self.extract_features_from_model(model_path, label=1, model_type='injected')
                    if features:
                        all_features.append(features)
                else:
                    print(f"Model not found: {model_path}")
        
        # Convert to DataFrame
        if all_features:
            self.features_df = pd.DataFrame(all_features)
            
            # Handle NaN values
            self.features_df = self.features_df.fillna(0)
            
            # Save features
            self._save_features()
            
            print(f"\n=== FEATURE EXTRACTION COMPLETED ===")
            print(f"Total samples: {len(self.features_df)}")
            print(f"Clean models: {len(self.features_df[self.features_df['label'] == 0])}")
            print(f"Injected models: {len(self.features_df[self.features_df['label'] == 1])}")
            print(f"Total features: {len(self.features_df.columns) - 4}")  # Exclude metadata columns
            
            return self.features_df
        else:
            print("No features extracted!")
            return None
    
    def _save_features(self):
        """Save features dataset"""
        os.makedirs('data/processed', exist_ok=True)
        
        # Save as CSV
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f'data/processed/steganalysis_features_{timestamp}.csv'
        self.features_df.to_csv(csv_path, index=False)
        print(f"Features saved to: {csv_path}")
        
        # Save as pickle (preserve data types)
        pkl_path = f'data/processed/steganalysis_features_{timestamp}.pkl'
        self.features_df.to_pickle(pkl_path)
        print(f"Features saved to: {pkl_path}")
        
        # Save feature description
        self._save_feature_description()
    
    def _save_feature_description(self):
        """Save feature descriptions"""
        feature_descriptions = {
            'global_mean': 'Mean of all weight values in the model',
            'global_std': 'Standard deviation of all weight values',
            'global_skewness': 'Skewness of weight distribution',
            'global_kurtosis': 'Kurtosis of weight distribution',
            'total_parameters': 'Total number of trainable parameters',
            'global_lsb_bias': 'Bias in LSB bits across all weights',
            'global_lsb_entropy': 'Entropy of LSB bit distribution',
            'avg_layer_correlation': 'Average correlation between layer weights',
            'max_layer_correlation': 'Maximum correlation between layer weights',
            'min_layer_correlation': 'Minimum correlation between layer weights',
            'std_layer_correlation': 'Standard deviation of layer correlations',
            'conv_weights_mean': 'Mean of convolutional layer weights',
            'conv_weights_std': 'Standard deviation of convolutional layer weights',
            'fc_weights_mean': 'Mean of fully connected layer weights',
            'fc_weights_std': 'Standard deviation of fully connected layer weights'
        }
        
        # Add layer-wise feature descriptions
        for col in self.features_df.columns:
            if col not in feature_descriptions:
                if '_mean' in col:
                    feature_descriptions[col] = f'Mean of weights in {col.replace("_mean", "")}'
                elif '_std' in col:
                    feature_descriptions[col] = f'Standard deviation of weights in {col.replace("_std", "")}'
                elif '_lsb_' in col:
                    feature_descriptions[col] = f'LSB analysis feature for {col.split("_lsb_")[0]}'
                elif '_entropy' in col:
                    feature_descriptions[col] = f'Entropy feature for {col.replace("_entropy", "")}'
        
        description_path = 'data/processed/feature_descriptions.yaml'
        with open(description_path, 'w') as f:
            yaml.dump(feature_descriptions, f, default_flow_style=False)
        
        print(f"Feature descriptions saved to: {description_path}")
    
    def get_feature_matrix(self):
        """Dapatkan feature matrix dan labels untuk training"""
        if self.features_df is None:
            print("No features extracted yet. Run extract_features_from_all_models() first.")
            return None, None
        
        # Exclude metadata columns
        exclude_cols = ['model_path', 'model_name', 'model_type', 'label']
        if 'injection_payload_type' in self.features_df.columns:
            exclude_cols.extend(['injection_payload_type', 'injection_lsb_bits', 'injection_ratio'])
        
        feature_cols = [col for col in self.features_df.columns if col not in exclude_cols]
        
        X = self.features_df[feature_cols].values
        y = self.features_df['label'].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Label distribution: {np.bincount(y)}")
        
        return X, y, feature_cols
    
    def analyze_feature_importance(self):
        """Analisis awal importance fitur"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X, y, feature_names = self.get_feature_matrix()
        
        if X is None:
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest untuk feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n=== TOP 20 MOST IMPORTANT FEATURES ===")
        print(importance_df.head(20))
        
        # Save feature importance
        importance_path = 'data/processed/feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        print(f"Feature importance saved to: {importance_path}")
        
        return importance_df

def create_feature_config():
    """Create feature extraction configuration file"""
    config = {
        'feature_extraction': {
            'weight_based_features': True,
            'activation_based_features': False,
            'statistical_moments': True,
            'lsb_analysis': True,
            'entropy_features': True,
            'batch_norm_features': True,
            'layer_wise_features': True,
            'correlation_features': True,
            'save_intermediate': True
        },
        'datasets': {
            'test_samples': 1000,
            'batch_size': 64
        }
    }
    
    os.makedirs('configs', exist_ok=True)
    with open('configs/feature_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Feature config created: configs/feature_config.yaml")

if __name__ == "__main__":
    # Create config file jika belum ada
    if not os.path.exists("configs/feature_config.yaml"):
        create_feature_config()
    
    # Test feature extraction
    extractor = FeatureExtractor()
    
    # Extract features from all models
    features_df = extractor.extract_features_from_all_models()
    
    if features_df is not None:
        # Analyze feature importance
        extractor.analyze_feature_importance()
        
        print("\n=== FEATURE EXTRACTION PIPELINE COMPLETED ===")
        print("Next step: Train detection models using the extracted features")