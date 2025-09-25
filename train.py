import torch
from tqdm import tqdm
from config import config
from utils.model_select import create_model
from sklearn.metrics import precision_score, recall_score
import os
from utils.generate_dataset import ProgressiveImageFolderDataset


def train_and_validate(model, data_module, criterion, optimizer, device, epochs, save_path="best_model.pth"):
    model.to(device)
    history = {
        'train_loss': [], 'train_acc': [],
        'valid_loss': [], 'valid_acc': [],
        'valid_precision': [], 'valid_recall': []
    }

    best_acc = 0.0
    best_precision = 0.0

    for epoch in range(epochs):
        # -------- Progressive unlock --------
        data_module.set_epoch(epoch)
        train_loader = data_module.train_dataloader()
        valid_loader = data_module.valid_dataloader()

        # -------- Train --------
        model.train()
        running_loss, correct, total = 0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training", leave=True)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_bar.set_postfix(loss=loss.item())

        train_loss = running_loss / total
        train_acc = correct / total

        # -------- Validation --------
        model.eval()
        running_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        valid_bar = tqdm(valid_loader, desc=f"Epoch [{epoch+1}/{epochs}] Validation", leave=True)
        with torch.no_grad():
            for inputs, labels in valid_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                valid_bar.set_postfix(loss=loss.item())

        valid_loss = running_loss / total
        valid_acc = correct / total
        valid_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        valid_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        # -------- Save Best --------
        if valid_acc > best_acc or valid_precision > best_precision:
            print(f" -------- model weights are updated! --------")
            torch.save(model.state_dict(), save_path)
            best_acc = max(best_acc, valid_acc)
            best_precision = max(best_precision, valid_precision)

        # -------- Record --------
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        history['valid_precision'].append(valid_precision)
        history['valid_recall'].append(valid_recall)

        # -------- Summary --------
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  Valid loss: {valid_loss:.4f}, acc: {valid_acc:.4f}")
        print(f"  Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}")

    return history

def main():
    device = config.device
    print("using {} device.".format(device))

    for model_name in config.model_list:
        # Dynamic dataset splitting
        data_module = ProgressiveImageFolderDataset(
            image_path=config.image_path,
            img_size=(config.image_size, config.image_size),
            batch_size=config.training_batch_size,
            valid_ratio=0.2,
            seed=42,
            milestones=[10, 15, 20, 25]
        )

        model = create_model(model_name, config.num_classes, continue_training=config.continue_weights).to(device)
        os.makedirs(config.save_path, exist_ok=True)
        criterion = config.training_loss
        optimizer = config.get_optimizer(model)
        epochs = config.training_epoch

        history = train_and_validate(
            model, data_module,
            criterion, optimizer, device, epochs,
            save_path=os.path.join(config.save_path, f"best_{model_name}_model.pth")
        )

        from utils.plot import save_history_json
        save_history_json(history, os.path.join(config.save_path, f'{model_name}_history.json'))


if __name__ == "__main__":
    main()