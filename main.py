import argparse
from src.data_acquisition import ModelAcquisition
from src.model_training import generate_all_model_variants
from src.injection_engine import LSBInjector
from src.feature_extractor import FeatureExtractor
from src.detector_trainer import DetectorTrainer
import os

def main():
    parser = argparse.ArgumentParser(description="Steganography Detection Research Project")
    parser.add_argument('--phase', type=str, required=True, 
                    choices=['acquire', 'train', 'inject', 'extract', 'detect'])
    parser.add_argument('--config', type=str, default='configs/base_config.yaml')
    
    args = parser.parse_args()
    
    if args.phase == 'acquire':
        print("=== PHASE 1: Data Acquisition ===")
        acquisition = ModelAcquisition(args.config)
        
        # Download model pre-trained
        pretrained_models = acquisition.get_pretrained_models()
        print(f"Downloaded {len(pretrained_models)} pre-trained models")
        
        # Download datasets
        datasets = acquisition.prepare_datasets()
        print(f"Prepared {len(datasets)} datasets")
    
    elif args.phase == 'train':
        print("=== PHASE 2: Model Training ===")
        all_models = generate_all_model_variants()
        print(f"Generated {len(all_models)} model variants")
    
    elif args.phase == 'inject':
        print("=== PHASE 3: LSB Injection ===")
        injector = LSBInjector(args.config)
        
        # Dapatkan semua model yang akan diinjeksi
        model_paths = []
        for root, dirs, files in os.walk("models/trained_models/"):
            for file in files:
                if file.endswith('.pth'):
                    model_paths.append(os.path.join(root, file))
        
        # Lakukan injeksi pada semua model
        for model_path in model_paths:
            print(f"Injecting LSB into {model_path}")
            injector.inject_lsb_to_model(model_path)
    
    elif args.phase == 'extract':
        print("=== PHASE 4: Feature Extraction ===")
        extractor = FeatureExtractor(args.config)
        extractor.extract_features_from_all_models()
    
    elif args.phase == 'detect':
        print("=== PHASE 5: Detector Training ===")
        trainer = DetectorTrainer(args.config)
        trainer.train_detector()
        trainer.evaluate_detector()

if __name__ == "__main__":
    main()