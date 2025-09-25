"""
Create a held-out test dataset from CT_meta (percentage split) and run model evaluation.
"""
import os
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import json
from datetime import datetime

from config import config


class TestDatasetGenerator:
    """Generate a test dataset by copying a percentage of images per class."""
    def __init__(self, source_path, test_path, test_ratio=0.1, seed=42):
        self.source_path = source_path
        self.test_path = test_path
        self.test_ratio = test_ratio
        self.seed = seed
    
    def create_test_dataset(self):
        """Create test dataset folder structure and copy sampled files."""
        if os.path.exists(self.test_path):
            shutil.rmtree(self.test_path)
        os.makedirs(self.test_path, exist_ok=True)
        
        print(f"[TestSet] Source: {self.source_path}")
        print(f"[TestSet] Ratio: {self.test_ratio}")
        
        class_folders = [f for f in os.listdir(self.source_path) \
                         if os.path.isdir(os.path.join(self.source_path, f))]
        
        total_copied = 0
        for class_name in class_folders:
            source_class_path = os.path.join(self.source_path, class_name)
            test_class_path = os.path.join(self.test_path, class_name)
            os.makedirs(test_class_path, exist_ok=True)
            
            all_files = [f for f in os.listdir(source_class_path) \
                         if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            np.random.seed(self.seed)
            n_test = max(1, int(len(all_files) * self.test_ratio))
            test_files = np.random.choice(all_files, n_test, replace=False)
            
            for file_name in test_files:
                shutil.copy2(os.path.join(source_class_path, file_name),
                             os.path.join(test_class_path, file_name))
            total_copied += len(test_files)
            print(f"  - {class_name}: {len(test_files)} files copied")
        
        print(f"[TestSet] Total copied files: {total_copied}")
        return self.test_path


class ModelEvaluator:
    """Evaluate a classification model on a test dataset."""
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def create_test_dataloader(self, test_path, batch_size=32, img_size=224):
        """Build DataLoader for test dataset."""
        test_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        test_dataset = datasets.ImageFolder(test_path, transform=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader, test_dataset.classes
        
    def predict(self, dataloader):
        """Run forward pass for entire dataloader and collect outputs."""
        all_predictions, all_probabilities, all_labels = [], [], []
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Predicting"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return (np.array(all_predictions),
                np.array(all_probabilities),
                np.array(all_labels))
    
    def calculate_metrics(self, y_true, y_pred, y_prob, class_names=None):
        """Compute evaluation metrics including specificity for binary case."""
        if class_names is None:
            class_names = ['Healthy', 'Tumor']
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        if len(class_names) == 2:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
            auc_roc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            specificity = None
            sensitivity = recall
            auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'auc_roc': auc_roc,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'class_names': class_names
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path=None):
        """Plot and optionally save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return cm
    
    def plot_roc_curve(self, y_true, y_prob, save_path=None):
        """Plot ROC curve for binary classification."""
        if y_prob.shape[1] != 2:
            return
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc_score = roc_auc_score(y_true, y_prob[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_metrics_report(self, metrics, save_path):
        """Save metrics to JSON + human readable TXT report (English)."""
        json_path = save_path.replace('.txt', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in metrics.items()}, f, indent=4)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('=' * 60 + '\n')
            f.write('CT Brain Tumor Model Evaluation Report\n')
            f.write('=' * 60 + '\n')
            f.write(f'Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n')
            f.write('Overall Metrics:\n')
            f.write('-' * 30 + '\n')
            f.write(f'Accuracy:     {metrics['accuracy']:.4f}\n')
            f.write(f'Precision:    {metrics['precision']:.4f}\n')
            f.write(f'Recall:       {metrics['recall']:.4f}\n')
            f.write(f'F1-Score:     {metrics['f1_score']:.4f}\n')
            if metrics['specificity'] is not None:
                f.write(f'Specificity:  {metrics['specificity']:.4f}\n')
            f.write(f'Sensitivity:  {metrics['sensitivity']:.4f}\n')
            f.write(f'AUC-ROC:      {metrics['auc_roc']:.4f}\n\n')
            f.write('Per-Class Metrics:\n')
            f.write('-' * 30 + '\n')
            for i, cname in enumerate(metrics['class_names']):
                f.write(f'{cname}:\n')
                f.write(f'  Precision: {metrics['precision_per_class'][i]:.4f}\n')
                f.write(f'  Recall:    {metrics['recall_per_class'][i]:.4f}\n')
                f.write(f'  F1-Score:  {metrics['f1_per_class'][i]:.4f}\n\n')


def load_model(model_name, model_path, num_classes=2):
    """Load a trained model by name and checkpoint path."""
    lname = model_name.lower()
    if lname == 'rnvit':
        from model.rnvit_model import RNViT
        model = RNViT(num_classes=num_classes)
    elif lname == 'vit':
        from model.vit_model import ViT
        model = ViT(num_classes=num_classes)
    elif lname == 'resnet':
        from model.resnet_model import ResNet
        model = ResNet(num_classes=num_classes)
    else:
        raise ValueError(f'Unknown model type: {model_name}')
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model


def main():
    """Main entry: generate test set (if needed) and evaluate model."""
    SOURCE_PATH = config.image_path
    TEST_PATH = os.path.join(os.getcwd(), 'CT_test')
    TEST_RATIO = 0.1
    SEED = 42
    MODEL_NAME = 'RNViT'  # 'vit', 'resnet'
    MODEL_PATH = r'PUT_YOUR_MODEL_PATH_HERE.pth'
    RESULTS_DIR = os.path.join(os.getcwd(), 'evaluation_results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print('=' * 60)
    print('CT Brain Tumor Evaluation')
    print('=' * 60)
    # Step 1: Create / refresh test dataset
    print('\n[Step 1] Creating test dataset subset')
    test_generator = TestDatasetGenerator(SOURCE_PATH, TEST_PATH, TEST_RATIO, SEED)
    test_path = test_generator.create_test_dataset()
    # Step 2: Load model
    print('\n[Step 2] Loading model')
    if not os.path.exists(MODEL_PATH):
        print(f'! Model file not found: {MODEL_PATH}')
        print('Provide a valid trained model checkpoint before evaluation.')
        return
    try:
        model = load_model(MODEL_NAME, MODEL_PATH, num_classes=config.num_classes)
        print(f'Loaded model: {MODEL_NAME}')
    except Exception as e:
        print(f'Failed to load model: {e}')
        return
    # Step 3: Evaluate
    print('\n[Step 3] Evaluating model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    evaluator = ModelEvaluator(model, device)
    test_loader, class_names = evaluator.create_test_dataloader(
        test_path,
        batch_size=config.training_batch_size,
        img_size=config.image_size
    )
    print(f'Classes: {class_names}')
    print(f'Test samples: {len(test_loader.dataset)}')
    predictions, probabilities, true_labels = evaluator.predict(test_loader)
    print('\n[Step 4] Computing metrics')
    metrics = evaluator.calculate_metrics(true_labels, predictions, probabilities, class_names)
    print('\n' + '=' * 60)
    print('Evaluation Summary')
    print('=' * 60)
    print(f'Accuracy:    {metrics['accuracy']:.4f}')
    print(f'Precision:   {metrics['precision']:.4f}')
    print(f'Recall:      {metrics['recall']:.4f}')
    print(f'F1-Score:    {metrics['f1_score']:.4f}')
    if metrics['specificity'] is not None:
        print(f'Specificity: {metrics['specificity']:.4f}')
    print(f'Sensitivity: {metrics['sensitivity']:.4f}')
    print(f'AUC-ROC:     {metrics['auc_roc']:.4f}')
    print(f"\n[Step 5] Saving outputs -> {RESULTS_DIR}")
    report_path = os.path.join(RESULTS_DIR, f'evaluation_report_{timestamp}.txt')
    evaluator.save_metrics_report(metrics, report_path)
    print(f'  - Report saved: {report_path}')
    cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_{timestamp}.png')
    evaluator.plot_confusion_matrix(true_labels, predictions, class_names, cm_path)
    print(f'  - Confusion matrix saved: {cm_path}')
    if len(class_names) == 2:
        roc_path = os.path.join(RESULTS_DIR, f'roc_curve_{timestamp}.png')
        evaluator.plot_roc_curve(true_labels, probabilities, roc_path)
        print(f'  - ROC curve saved: {roc_path}')
    print('\n Done. All results saved.')


if __name__ == '__main__':
    main()
