"""
Script to download the BreastMNIST dataset.

Run this once before running main.py
"""

import os

def download_with_medmnist():
    """Download using the medmnist package."""
    try:
        import medmnist
        from medmnist import BreastMNIST
        import numpy as np

        print("Downloading BreastMNIST using medmnist package...")

        # Create directory
        os.makedirs('Datasets/BreastMNIST', exist_ok=True)

        # Download datasets
        train_dataset = BreastMNIST(split='train', download=True, root='Datasets')
        val_dataset = BreastMNIST(split='val', download=True, root='Datasets')
        test_dataset = BreastMNIST(split='test', download=True, root='Datasets')

        # Save as .npz file in expected format
        npz_path = 'Datasets/BreastMNIST/breastmnist.npz'
        np.savez(npz_path,
                 train_images=train_dataset.imgs,
                 train_labels=train_dataset.labels,
                 val_images=val_dataset.imgs,
                 val_labels=val_dataset.labels,
                 test_images=test_dataset.imgs,
                 test_labels=test_dataset.labels)

        print(f"Dataset saved to {npz_path}")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val:   {len(val_dataset)} samples")
        print(f"  Test:  {len(test_dataset)} samples")
        return True

    except ImportError:
        print("medmnist package not installed. Install with: pip install medmnist")
        return False


def download_with_urllib():
    """Download directly from Zenodo."""
    import urllib.request

    url = "https://zenodo.org/records/10519652/files/breastmnist.npz"
    save_path = "Datasets/BreastMNIST/breastmnist.npz"

    os.makedirs('Datasets/BreastMNIST', exist_ok=True)

    print(f"Downloading from {url}...")
    print("This may take a moment...")

    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Dataset saved to {save_path}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


if __name__ == "__main__":
    print("="*50)
    print("BreastMNIST Dataset Downloader")
    print("="*50)

    # Try medmnist first, then direct download
    if not download_with_medmnist():
        print("\nTrying direct download from Zenodo...")
        download_with_urllib()

    print("\nDone! You can now run: python main.py")
