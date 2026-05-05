import argparse
import os

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import get_val_transforms
from model import DeForge_AI_Model


class BenchmarkDataset(Dataset):
    def __init__(self, hf_data, transform=None):
        self.hf_data = hf_data
        self.transform = transform

    def __len__(self):
        return len(self.hf_data)

    def __getitem__(self, idx):
        item = self.hf_data[idx]
        image = item['image'].convert('RGB')
        label = item['label']

        # In AIGC-Detection-Benchmark: 0 is Real, 1-17 are Fakes
        # Our model expects: 0 for Real, 1 for Fake
        target = 0.0 if label == 0 else 1.0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(target, dtype=torch.float32)


def run_test(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    print('Running evaluation on the full test set...')
    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            preds = torch.sigmoid(logits).squeeze(1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(
        description='Test DeForge-AI on full AIGC-Detection-Benchmark'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/model_epoch_best.pth',
        help='Path to model checkpoint',
    )
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument(
        '--limit', type=int, default=None, help='Limit total number of samples to test'
    )
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if not os.path.exists(args.checkpoint):
        print(f'Error: Checkpoint {args.checkpoint} not found.')
        return

    print(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    checkpoint_args = checkpoint.get('args', {})

    model_kwargs = {
        'lora_r': checkpoint_args.get('lora_r', 16),
        'lora_alpha': checkpoint_args.get('lora_alpha', 32),
        'lora_dropout': checkpoint_args.get('lora_dropout', 0.5),
        'unfreeze_last_blocks': checkpoint_args.get('unfreeze_last_blocks', 0),
        'image_size': checkpoint_args.get('image_size', args.image_size),
        'forensic_dim': checkpoint_args.get('forensic_dim', 256),
    }

    lora_target_modules = checkpoint_args.get('lora_target_modules')
    if isinstance(lora_target_modules, str):
        model_kwargs['lora_target_modules'] = [
            m.strip() for m in lora_target_modules.split(',') if m.strip()
        ]
    elif lora_target_modules:
        model_kwargs['lora_target_modules'] = lora_target_modules

    model = DeForge_AI_Model(**model_kwargs).to(device)
    model.load_state_dict(
        checkpoint['model_state_dict']
        if 'model_state_dict' in checkpoint
        else checkpoint,
        strict=False,
    )

    print('Loading AIGC-Detection-Benchmark dataset...')
    dataset = load_dataset(
        'TheKernel01/AIGC-Detection-Benchmark', split='test', token=hf_token
    )
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    test_ds = BenchmarkDataset(
        dataset, transform=get_val_transforms(size=args.image_size)
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    preds, labels = run_test(model, test_loader, device)

    # Calculate metrics
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
    auroc = metrics.auc(fpr, tpr)
    ap = metrics.average_precision_score(labels, preds)

    binary_preds = (preds > 0.5).astype(float)
    acc = (binary_preds == labels).mean()

    real_mask = labels == 0
    fake_mask = labels == 1
    real_acc = (
        (binary_preds[real_mask] == labels[real_mask]).mean() if real_mask.any() else 0
    )
    fake_acc = (
        (binary_preds[fake_mask] == labels[fake_mask]).mean() if fake_mask.any() else 0
    )

    print('\n' + '=' * 40)
    print('Overall Results (Full Test Set)')
    print('-' * 40)
    print(f'Total Samples:    {len(labels)}')
    print(f'Overall Accuracy: {acc:.4f}')
    print(f'Real Accuracy:    {real_acc:.4f}')
    print(f'Fake Accuracy:    {fake_acc:.4f}')
    print(f'Balanced Acc:    {(real_acc + fake_acc) / 2:.4f}')
    print(f'AUC:             {auroc:.4f}')
    print(f'AP:              {ap:.4f}')
    print('=' * 40)


if __name__ == '__main__':
    main()
