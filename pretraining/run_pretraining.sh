#!/bin/bash

#PBS -N run_pretraining
#PBS -V
#PBS -q preemptable
#PBS -l select=2:system=polaris
#PBS -l walltime=40:00:00
#PBS -l filesystems=home:grand
#PBS -A SolarWindowsADSP
#PBS -e /grand/SolarWindowsADSP/alanyahya/run_pretraining_e.sh
#PBS -o /grand/SolarWindowsADSP/alanyahya/run_pretraining_o.sh
#PBS -m be
#PBS -r y

HF_HOME=/grand/SolarWindowsADSP/alanyahya/.cache/huggingface
export HF_HOME=$HF_HOME

export http_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export https_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export ftp_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"

export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export CRAY_ACCEL_TARGET=nvidia80

echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

. /etc/profile

module use /soft/modulefiles; module load conda; conda activate base; python /grand/SolarWindowsADSP/alanyahya/hflogin.py
echo python3: $(which python3)

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job ID: ${PBS_JOBID}"
echo "Job started at: ${TSTAMP}"

DS_HOSTFILE="./hostfile"
echo "DS_HOSTFILE: ${DS_HOSTFILE}"
#here we are creating a hostfile with the list of nodes
DS_ENVFILE="./.deepspeed_env"
echo "DS_ENVFILE: ${DS_ENVFILE}"
#here we are creating a file with the environment variables

NRANKS=$(wc -l < "${PBS_NODEFILE}")
NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
NGPUS="$((${NRANKS}*${NGPU_PER_RANK}))"
echo "NRANKS: ${NRANKS}, NGPU_PER_RANK: ${NGPU_PER_RANK}, NGPUS: ${NGPUS}"

cat "${PBS_NODEFILE}" > "${DS_HOSTFILE}"
sed -e 's/$/ slots=4/' -i "${DS_HOSTFILE}"

echo "Writing environment variables to: ${DS_ENVFILE}"
echo "PATH=${PATH}" > "${DS_ENVFILE}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> "${DS_ENVFILE}"
echo "https_proxy=${https_proxy}" >> "${DS_ENVFILE}"
echo "http_proxy=${http_proxy}" >> "${DS_ENVFILE}"

most_recent_checkpoint=$(python /grand/SolarWindowsADSP/alanyahya/NF_NanoBERT_V4.py) 

deepspeed \
    --hostfile="${DS_HOSTFILE}" \
    /grand/SolarWindowsADSP/alanyahya/run_mlm-from-pretokenized.py \
        --model_name_or_path google-bert/bert-base-uncased \
        --trust_remote_code True \
        --dataset_name Flamenco43/tokenized_1.8b_nano \
        --max_seq_length 512 \
        --mlm_probability 0.15 \
        --bf16 True \
        --bf16_full_eval True \
        --do_train \
        --do_eval \
        --eval_strategy steps \
        --save_strategy steps \
        --save_steps 1000 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --deepspeed /grand/SolarWindowsADSP/alanyahya/ds_z3.json \
        --learning_rate 5e-5 \
        --max_steps 500000 \
        --include_tokens_per_second True \
        --auto_find_batch_size False \
        --output_dir /grand/SolarWindowsADSP/alanyahya/models/NanoBERT_V4 \
        --run_name NanoBERT_V4 \
        --push_to_hub \
        --hub_private_repo True \
        --resume_from_checkpoint ${most_recent_checkpoint} \