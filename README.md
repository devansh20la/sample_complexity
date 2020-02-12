# Simple and Accurate Approach for Estimating Sample Complexity of Deep Neural Networks
The aim of the project is to estimate practical bounds for sample complexity in deep neural networks. 

## Prerequisites
What things you need to install the software and how to install them
```
pytorch
torchvision
PIL
sklearn
scikit-learn
tensorboard (for logging metrics)
tqdm (for pretty progress bars)

```

## Training
```
--ms: manual seed (eg 123)
--n: name for saving all the files (The results are saved in args.dir \ "checkpoints" \ args.name)

--img_per_class: images per class (used for imagenet) e.g 1000, 500 etc
--train_size: size of training data in % (used for all other data types) e.g 0.75, 0.25, 0.5 etc

--print_freq: frequency to print statistics
--mtype: model type
	choices are:
		"cifar_resnet18"
		"cifar_vgg16"
		"cifar_resnet50"
		"imagenet_resnet50"
		"lenet"
		"udacity_cnn"

--dtype: data type
	choices are:
		"cifar10"
		"cifar100"
		"mnist"
		"imagenet"
		"udacity"

--ep: total epochs 
--opt: optimizer
	"sgd"
	"adam"
--lr: learning rate
--wd: weight decay
--bs: batch size
--m: momentum
```
An example to train on cifar 10 data with 100% training data on resnet 18 model is as follows:
```
python train.py --dtype="cifar10" --train_size=1.0 \
 	--mtype="cifar_resnet18" --ep=500 --ms=123 --opt="sgd" --wd=1e-4 \
 	--lr=0.1 --bs=128 --print_freq=100 \
 	--n="cifar10_resnet18/cifar10_1.0/"
```
## Running the tests
An example to test on cifar 10 test data on resnet 18 model is as follows:
```
python test.py --dtype="cifar10" --modeltype="cifar_resnet18" \
	--bs=128 --cp_dir="checkpoints/cifar10_resnet18/"
```
The results are saved in the folder "results/cifar10/"

## Authors



## License


## Acknowledgments

