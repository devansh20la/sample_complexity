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
for ms in 10 20 30 40 50
do
	for lp in 1.0 0.75 0.5 0.25 0.125 0.0625 0.03125
	do
		get_device
		echo "Running on Device: ${device}"
		CUDA_VISIBLE_DEVICES=$((device)) nohup python train.py \
			--dtype=${dtype} \
			--train_size=${lp} \
			--mtype=${mtype}  \
			--ep=${ep} \
			--ms=${ms} \
			--opt=${opt} \
			--wd=${wd} \
			--lr=${lr} \
			--bs=${bs} \
			--print_freq=100 &
		sleep 2
	done
done
