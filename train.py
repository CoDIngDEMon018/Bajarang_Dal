EPOCHS = 8
MOSAIC = 0.1
OPTIMIZER = 'AdamW'
MOMENTUM = 0.2
LR0 = 0.001
LRF = 0.0001
SINGLE_CLS = False
import argparse
from ultralytics import YOLO
import os
import sys

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    # epochs
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    # mosaic
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    # optimizer
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    # momentum
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    # lr0
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    # lrf
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    # single_cls
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')
    args = parser.parse_args()
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    model = YOLO(os.path.join(this_dir, "yolov8s.pt"))
    results = model.train(
            data=os.path.join(this_dir, "yolo_params.yaml"),
        
            epochs=8,  # Extended training
            imgsz=1000,  # Higher resolution
            batch=4,      # Adjust for GPU memory
            lr0=0.001,
            lrf=0.00001,  # Very slow decay
            optimizer='AdamW',
            weight_decay=0.1,  # Strong regularization
            box=12.0,     # Increased box loss weight
            cls=0.5,      # Reduced cls weight
            dfl=3.0,      # Emphasize distribution focal loss
            label_smoothing=0.1,  # Prevent overconfidence
            mosaic=0.9,
            mixup=0.15,
            copy_paste=0.2,
            hsv_h=0.02,   # Aggressive color aug
            hsv_s=0.8,
            hsv_v=0.4,
            degrees=15.0,  # Geometric aug
            shear=5.0,
            perspective=0.001,
            fliplr=0.5,
            auto_augment='randaugment',
            erasing=0.4,
            nbs=128,       # Normalized batch size
            overlap_mask=True,
            mask_ratio=8,  # Higher mask resolution
            # anchor_t=3.0,  # Tighter anchor matching
            close_mosaic=15,
            amp=True,
            patience=50,   # Early stopping
            plots=True,
    )

#     results = model.train(
#     data=os.path.join(this_dir, "yolo_params.yaml"),
#     epochs=10,  # Increased from 5
#     imgsz=1024,  # Higher resolution
#     batch=4,  # Reduced for higher resolution
#     lr0=0.001,
#     lrf=0.0001,
#     optimizer='AdamW',
#     weight_decay=0.05,  # Stronger regularization
#     box=9.0,  # Increased box loss weight
#     cls=0.8,   # Reduced cls loss weight
#     dfl=2.0,   # Increased distribution focal loss
#     mosaic=0.8,
#     mixup=0.1,  # Reduced mixup
#     copy_paste=0.05,
#     close_mosaic=10,
#     degrees=5.0,  # Reduced rotation
#     shear=1.0,    # Reduced shear
#     perspective=0.0005,
#     fliplr=0.3,
#     auto_augment='randaugment',
#     # anchor_t=4.0,  # Tighter anchor matching
#     nbs=64,        # Normalized batch size
#     overlap_mask=True,
#     mask_ratio=4,
#     single_cls=False,
#     amp=True,      # Keep mixed precision
#     patience=30,   # Early stopping
#     plots=True
# )
    
    import torch
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
'''
Mixup boost val pred but reduces test pred
Mosaic shouldn't be 1.0  
'''


'''
                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]
Model summary: 225 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs
'''