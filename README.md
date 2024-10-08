# A Theoretical-Empirical Approach to Estimating Sample Complexity of DNNs
The aim of the project is to estimate practical bounds for sample complexity for deep neural networks. 

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
## Simulation (learning_curve/simulation)
The simulation codes are provided in two parts (a) simulate_1d.py for 1D dimensional data and (b) simulate_hd.py for high dimensional data. 

## Empirical learning curve
The empirical learning curve can be generated by following steps:
1) Fix train dataset, test dataset and model.
2) Train multiple copies of the model on different training data sizes.
3) Test each model on fixed test set. This will generate an empirical learning curve.

### (Step 2) Training (train_dnn/train_model)
```
--ms: manual seed (eg 123)

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
Example to train on cifar 10 data with 100% training data on resnet 18 model:
```
python train.py \
	--dtype="cifar10" \
	--train_size=1.0 \
	--mtype="cifar_resnet18"  \
	--ep=500 \
	--ms=123 \
	--opt="sgd" \
	--wd=1e-4 \
	--lr=0.1 \
	--bs=128 \
	--print_freq=100 &
```

### (Step 3) Testing (train_dnn/train_model)
Example to test on cifar 10 test data on resnet 18 model:
```
python test.py --dtype="cifar10" --modeltype="cifar_resnet18" \
	--bs=128 --cp_dir="checkpoints/cifar10_resnet18/"
```
The empirical learning curve data will saved in the folder "results/cifar10/"

## Estimating theoretical learning curve
Steps to estimate future learning curve is as follows:
1) Generate feature maps for trained model.
2) Generate theoretical plot

### (Step 1) Saving feature maps (learning_curve/save_feature_maps)
Example to save features for cifar 10 data trained on resnet 18 model:
```
python save_feature_map.py --dtype="cifar10" --mtype="cifar_resnet18"
```

### (Step 2) Estimating the learning curve (learning_curve/estimate_learning_curve)
The following arguments are required
```
--runs :  multiple runs to compute standard deviation
--exp_results_path: path to empirical error file (generated from test.py)
--features_path: path to features generated from (save_features_map.py)
--mtype: modeltype
--dtype: datatype
```
Example on cifar10 dataset with:
```
python fit_curve.py --run=10 --dtype="cifar10" --mtype="cifar_resnet18" --exp_results_path="" --features_path=""
```

## Additional experiments
## Bottleneck (train_dnn/train_bottleneck)
### Training
Bottleckneck model requires same set of paramters as previous train.py with some additional parameters mentioned below
```
--dim = dimension 
--load_cp=load model state from previosly saved file
```

Example to train bottleneck model with dimension 2 on cifar 10 data:
```
python bottleneck_train.py \
	--dtype="cifar10" \
	--train_size=1.0 \
	--mtype="cifar_resnet18"  \
	--ep=500 \
	--ms=50 \
	--opt="sgd" \
	--wd=1e-4 \
	--lr=0.1 \
	--bs=128 \
	--dim=2 \
	--print_freq=100 \
	--load_cp="../train_model/checkpoints/cifar10/cifar_resnet18/cifar_resnet18_${train_size}/run0/" &
```
### Testing
```
python bottleneck_test.py --mtype="cifar_resnet18" --dtype="cifar10"
```

## Nearest neighbors (train_dnn/train_bottleneck/k_nn.py)
Firstly, we save features from bottleneck to run nearest neighbor. 
### Save feature maps
Example to run save_feature_maps.py on mnist datasets is:
```
python save_feature_maps.py --dtype="mnist" --mtype="lenet"
```
### NN
Following arguments are requried:
```
--data_path: path to saved feature maps
--dtype: data type
--mtype: model type
--nn: nearest neigbour
```
Example to run k_nn.py on mnist datasets is:
```
python k_nn.py --data_path="data/mnist/bottleneck_mnist_lenet_feat.hdf5" --dtype="mnist" --mtype="lenet" --nn=1 
```
## Authors
