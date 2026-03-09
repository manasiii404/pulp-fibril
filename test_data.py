import sys
from data_pipeline.dataset import get_dataloaders
import torch

def test_loader():
    with open('test_data_log_utf8.txt', 'w', encoding='utf-8') as f:
        sys.stdout = f
        
        train_loader, val_loader, test_loader = get_dataloaders(
            data_root="data/synthetic",
            image_size=256,
            batch_size=2,
            num_workers=0
        )
        
        batch = next(iter(train_loader))
        print(f"Images shape: {batch['images'].shape}")
        
        masks_list = batch['masks']
        labels_list = batch['labels']
        boxes_list = batch['boxes']
        
        for i in range(len(masks_list)):
            print(f"Batch {i}:")
            print(f"  Masks shape:  {masks_list[i].shape} - max value: {float(masks_list[i].max()) if len(masks_list[i]) > 0 else 'N/A'}")
            print(f"  Labels shape: {labels_list[i].shape}")
            print(f"  Boxes shape:  {boxes_list[i].shape}")
            
if __name__ == "__main__":
    test_loader()
