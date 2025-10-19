import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm
import os
from src.utils.helpers import HelperDataset

class ModelEvaluator:
    """Evaluator untuk model cover dan stego"""
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.helper_data_loader = helper.get_data_loader()
    
    def _save_results(self, results: List, evaluation_results: List):
        """Save semua hasil processing dan evaluasi"""
        import json
        import pandas as pd
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save processing summary
        summary_path = os.path.join('data/results', f"processing_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save evaluation results jika ada
        if evaluation_results:
            eval_path = os.path.join('data/results', f"evaluation_results_{timestamp}.json")
            
            # Simplify evaluation results untuk JSON
            simplified_eval = []
            for eval_result in evaluation_results:
                simplified = {
                    'model_file': eval_result['model_file'],
                    'x_bits': eval_result['x_bits'],
                    'dataset': eval_result['dataset'],
                    'architecture': eval_result['architecture'],
                    'cover_accuracy': eval_result['evaluation']['cover_model']['accuracy'],
                    'stego_accuracy': eval_result['evaluation']['stego_model']['accuracy'],
                    'accuracy_difference': eval_result['evaluation']['differences']['accuracy_difference'],
                    'accuracy_change_percent': eval_result['evaluation']['differences']['accuracy_change_percent']
                }
                simplified_eval.append(simplified)
            
            with open(eval_path, 'w') as f:
                json.dump(simplified_eval, f, indent=2)
            
            # Juga save sebagai CSV untuk analisis mudah
            csv_path = os.path.join('data/results', f"evaluation_results_{timestamp}.csv")
            df = pd.DataFrame(simplified_eval)
            df.to_csv(csv_path, index=False)
            
            print(f"\n📈 Evaluation results saved:")
            print(f"   JSON: {eval_path}")
            print(f"   CSV:  {csv_path}")
        
        print(f"\n📋 Processing summary saved: {summary_path}")

    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluasi model lengkap: accuracy, loss, confidence scores
        """
        model.eval()
        model.to(self.device)
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_confidences = []
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Evaluating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                # Accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Confidence scores (softmax probabilities)
                probabilities = torch.softmax(outputs, dim=1)
                max_probs, _ = probabilities.max(dim=1)
                all_confidences.extend(max_probs.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(test_loader)
        avg_confidence = np.mean(all_confidences)
        confidence_std = np.std(all_confidences)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'confidence_mean': avg_confidence,
            'confidence_std': confidence_std,
            'total_samples': total,
            'correct_predictions': correct
        }
    
    def compare_models(self, cover_model: nn.Module, stego_model: nn.Module, 
                    dataset_name: str, batch_size=128) -> Dict:
        """
        Bandingkan performa cover model vs stego model
        """
        print(f"\n{'='*60}")
        print(f"COMPARING MODELS on {dataset_name}")
        print(f"{'='*60}")
        helper = HelperDataset()
        
        test_loader = helper.get_data_loader(dataset_name, batch_size)
        
        # Evaluate both models
        cover_metrics = self.evaluate_model(cover_model, test_loader)
        stego_metrics = self.evaluate_model(stego_model, test_loader)
        
        # Calculate differences
        accuracy_diff = stego_metrics['accuracy'] - cover_metrics['accuracy']
        loss_diff = stego_metrics['loss'] - cover_metrics['loss']
        confidence_diff = stego_metrics['confidence_mean'] - cover_metrics['confidence_mean']
        
        comparison = {
            'cover_model': cover_metrics,
            'stego_model': stego_metrics,
            'differences': {
                'accuracy_difference': accuracy_diff,
                'loss_difference': loss_diff,
                'confidence_difference': confidence_diff,
                'accuracy_change_percent': (accuracy_diff / cover_metrics['accuracy']) * 100
            }
        }
        
        # Print results
        self._print_comparison(comparison, dataset_name)
        
        return comparison
    
    def _print_comparison(self, comparison: Dict, dataset_name: str):
        """Print hasil perbandingan"""
        cover = comparison['cover_model']
        stego = comparison['stego_model']
        diff = comparison['differences']
        
        print(f"\n📊 PERFORMANCE COMPARISON on {dataset_name}")
        print(f"{'Metric':<20} {'Cover':<10} {'Stego':<10} {'Difference':<12} {'Change %':<10}")
        print(f"{'-'*60}")
        
        print(f"{'Accuracy (%)':<20} {cover['accuracy']:<10.2f} {stego['accuracy']:<10.2f} "
            f"{diff['accuracy_difference']:<12.2f} {diff['accuracy_change_percent']:<10.2f}")
        
        print(f"{'Loss':<20} {cover['loss']:<10.4f} {stego['loss']:<10.4f} "
            f"{diff['loss_difference']:<12.4f} {'-':<10}")
        
        print(f"{'Confidence':<20} {cover['confidence_mean']:<10.4f} {stego['confidence_mean']:<10.4f} "
            f"{diff['confidence_difference']:<12.4f} {'-':<10}")
        
        # Interpret results
        accuracy_change = abs(diff['accuracy_difference'])
        if accuracy_change < 0.1:
            impact = "✅ NEGLIGIBLE"
        elif accuracy_change < 1.0:
            impact = "⚠️ MINOR" 
        elif accuracy_change < 5.0:
            impact = "🔶 MODERATE"
        else:
            impact = "🔴 SIGNIFICANT"
        
        print(f"\n📈 IMPACT ASSESSMENT: {impact}")
        print(f"   Accuracy change: {diff['accuracy_difference']:.4f}%")