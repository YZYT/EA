python train.py --dataset cifar100 --model resnet18 --data_augmentation --epochs 250

python train.py --dataset cifar100 --model resnet18 --data_augmentation  --lookahead --epochs 250 --pullback_momentum pullback

python train.py --SWA --epochs 250
python train.py --LA --epochs 250
python train.py  --epochs 250

# 

# python train.py --dataset cifar10 --model resnet18 --data_augmentation  --lookahead
# python train.py --dataset cifar10 --model resnet18 --data_augmentation
# python train.py --dataset cifar10 --model wideresnet --data_augmentation
# python train.py --dataset cifar10 --model wideresnet --data_augmentation --lookahead
# python train.py --dataset cifar100 --model wideresnet --data_augmentation
# python train.py --dataset cifar100 --model wideresnet --data_augmentation --lookahead



python train3.py --dir=swa --dataset=CIFAR100 --data_path=data --model=VGG16  --swa 