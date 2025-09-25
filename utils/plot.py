from matplotlib import pyplot as plt
import json
import os
import numpy as np
from datetime import datetime

def load_json_data(
    file_path: str,
):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: Data file not found {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: JSON parsing failed - {e}")
        return []


def save_history_json(history, filepath):
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=4)


def plot_history(history, model_name, save_path=None):
    """
    Plot training and validation process charts
    
    Args:
        history: Can be file path or direct history data dictionary
        model_name: Model name
        save_path: Path to save the plot, if None use default path
    """
    # If input is string, treat as file path
    if isinstance(history, str):
        history_data = load_json_data(history)
    else:
        history_data = history
        
    if not history_data:
        print("No history data found!")
        return
    
    # Check if required keys exist
    required_keys = ['train_loss', 'valid_loss', 'train_acc', 'valid_acc']
    missing_keys = [key for key in required_keys if key not in history_data]
    if missing_keys:
        print(f"Warning: Missing keys in history data: {missing_keys}")
        return
    
    epochs = range(1, len(history_data['train_loss']) + 1)
    
    # 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name} Training History', fontsize=16)
    
    # 1. Loss curves
    ax1.plot(epochs, history_data['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history_data['valid_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy curves
    ax2.plot(epochs, history_data['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history_data['valid_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision and Recall
    if 'valid_precision' in history_data and 'valid_recall' in history_data:
        ax3.plot(epochs, history_data['valid_precision'], 'g-', label='Validation Precision', linewidth=2)
        ax3.plot(epochs, history_data['valid_recall'], 'm-', label='Validation Recall', linewidth=2)
        if 'valid_f1' in history_data:
            ax3.plot(epochs, history_data['valid_f1'], 'c-', label='Validation F1-Score', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.set_title('Validation Metrics')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        # If no precision and recall, show detailed training accuracy information
        ax3.plot(epochs, np.array(history_data['train_acc']) * 100, 'b-', label='Training Accuracy (%)', linewidth=2)
        ax3.plot(epochs, np.array(history_data['valid_acc']) * 100, 'r-', label='Validation Accuracy (%)', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Accuracy Percentage')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Learning rate curve (if exists)
    if 'learning_rate' in history_data:
        ax4.plot(epochs, history_data['learning_rate'], 'orange', label='Learning Rate', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')  # Use logarithmic scale for learning rate display
    else:
        train_loss_smooth = np.convolve(history_data['train_loss'], np.ones(5)/5, mode='valid')
        valid_loss_smooth = np.convolve(history_data['valid_loss'], np.ones(5)/5, mode='valid')
        smooth_epochs = range(3, len(history_data['train_loss']) - 1)
        
        ax4.plot(epochs, history_data['train_loss'], 'b-', alpha=0.3, label='Training Loss (Raw)')
        ax4.plot(epochs, history_data['valid_loss'], 'r-', alpha=0.3, label='Validation Loss (Raw)')
        ax4.plot(smooth_epochs, train_loss_smooth, 'b-', linewidth=2, label='Training Loss (Smoothed)')
        ax4.plot(smooth_epochs, valid_loss_smooth, 'r-', linewidth=2, label='Validation Loss (Smoothed)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title('Loss Curves (Smoothed)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = os.path.join(os.getcwd(), 'training_plots')
    
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_training_history_{timestamp}.jpg"
    filepath = os.path.join(save_path, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {filepath}")
    
    # Optional: Show plot
    # plt.show()
    plt.close()
    
    return filepath


def plot_metrics_comparison(histories_dict, save_path=None):
    """
    Compare training histories of multiple models
    
    Args:
        histories_dict: Dictionary in format {model_name: history_data}
        save_path: Save path
    """
    if not histories_dict:
        print("No history data provided!")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Comparison', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, (model_name, history) in enumerate(histories_dict.items()):
        if isinstance(history, str):
            history = load_json_data(history)
        
        if not history:
            continue
            
        epochs = range(1, len(history['train_loss']) + 1)
        color = colors[i % len(colors)]
        
        # Training Loss
        ax1.plot(epochs, history['train_loss'], color=color, linestyle='-', 
                label=f'{model_name} (Train)', alpha=0.7)
        
        # Validation Loss
        ax2.plot(epochs, history['valid_loss'], color=color, linestyle='--', 
                label=f'{model_name} (Valid)', alpha=0.7)
        
        # Training Accuracy
        ax3.plot(epochs, history['train_acc'], color=color, linestyle='-', 
                label=f'{model_name} (Train)', alpha=0.7)
        
        # Validation Accuracy
        ax4.plot(epochs, history['valid_acc'], color=color, linestyle='--', 
                label=f'{model_name} (Valid)', alpha=0.7)
    
    # Set up subplots
    ax1.set_title('Training Loss Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Validation Loss Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('Training Accuracy Comparison')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('Validation Accuracy Comparison')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = os.path.join(os.getcwd(), 'training_plots')
    
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_comparison_{timestamp}.jpg"
    filepath = os.path.join(save_path, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to: {filepath}")
    plt.close()
    
    return filepath


def plot_confusion_matrix_from_history(history, model_name, save_path=None):
    """
    Plot confusion matrix from history data (if exists)
    """
    if isinstance(history, str):
        history = load_json_data(history)
    
    if 'confusion_matrix' not in history:
        print("No confusion matrix data found in history!")
        return
    
    import seaborn as sns
    
    # Get confusion matrix from the last epoch
    cm = np.array(history['confusion_matrix'][-1])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Tumor'], 
                yticklabels=['Healthy', 'Tumor'])
    plt.title(f'{model_name} - Final Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save plot
    if save_path is None:
        save_path = os.path.join(os.getcwd(), 'training_plots')
    
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_confusion_matrix_{timestamp}.jpg"
    filepath = os.path.join(save_path, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to: {filepath}")
    plt.close()
    
    return filepath


def create_training_summary_plot(history, model_name, save_path=None):
    """
    Create training summary plot with most important metrics
    """
    if isinstance(history, str):
        history = load_json_data(history)
    
    if not history:
        print("No history data found!")
        return
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create dual y-axis
    ax2 = ax.twinx()
    
    # Plot loss on left axis
    line1 = ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    line2 = ax.plot(epochs, history['valid_loss'], 'r-', linewidth=2, label='Validation Loss')
    
    # Plot accuracy on right axis
    line3 = ax2.plot(epochs, np.array(history['train_acc']) * 100, 'b--', linewidth=2, label='Training Accuracy (%)')
    line4 = ax2.plot(epochs, np.array(history['valid_acc']) * 100, 'r--', linewidth=2, label='Validation Accuracy (%)')
    
    # Set labels and title
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'{model_name} - Training Summary', fontsize=14)
    
    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add annotation for best validation accuracy
    best_valid_acc_epoch = np.argmax(history['valid_acc']) + 1
    best_valid_acc = max(history['valid_acc']) * 100
    ax2.annotate(f'Best: {best_valid_acc:.2f}% (Epoch {best_valid_acc_epoch})',
                xy=(best_valid_acc_epoch, best_valid_acc),
                xytext=(best_valid_acc_epoch + 10, best_valid_acc + 5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = os.path.join(os.getcwd(), 'training_plots')
    
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_training_summary_{timestamp}.jpg"
    filepath = os.path.join(save_path, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Training summary plot saved to: {filepath}")
    plt.close()
    
    return filepath


# Usage examples
if __name__ == "__main__":
    # Single model training history plotting
    plot_history(r"G:\CT-brain\checkpoints\2509241822\rn_vit_base_patch16_224_history.json", 'RNViT_Model')
    
    # Multi-model comparison
    # histories = {
    #     'RNViT': r"G:\CT-brain\checkpoints\2509241822\rn_vit_base_patch16_224_history.json",
    #     # 'ViT': r"G:\CT-brain\checkpoints\2509241822\vit_base_patch16_224_history.json",
    #     'ResNet': r"G:\CT-brain\checkpoints\2509241347\resnet50_history.json"
    # }
    # plot_metrics_comparison(histories)