#!/bin/bash
query_device() {
	for d in 0 1 2 3
	do
		dmem="$(nvidia-smi -q -i ${d} -d Memory |grep -A4 GPU | grep Used | grep -Eo '[0-9]{1,5}')"
		if (($dmem < 50)); then
		    device=${d}
		fi
	done
}

get_device() {
	unset device
	while [[ -z "${device}" ]]
	do
		query_device
		if [[ -z "${device}" ]]; then
			echo "All devices are busy, sleeping for 60s"
			sleep 60
		fi
	done
}

opt="sgd"
ep=1
wd=0.0005
bs=128
lr=0.1
dtype="mnist"
mtype="lenet"
train_size=1.0
load_cp="../train_model/checkpoints/mnist/lenet/lenet_${train_size}/run0/"
for dim in 1 2 3 4 5 6 7 
do
	get_device
	echo "Running on Device: ${device}"
	CUDA_VISIBLE_DEVICES=$((device)) nohup python bottleneck_train.py \
		--dtype=${dtype} \
		--train_size=${train_size} \
		--mtype=${mtype}  \
		--ep=${ep} \
		--ms=50 \
		--opt=${opt} \
		--wd=${wd} \
		--lr=${lr} \
		--bs=${bs} \
		--dim=${dim} \
		--print_freq=100 \
		--load_cp=${load_cp} &
	sleep 2
done

