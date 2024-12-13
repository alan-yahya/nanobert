#!/bin/bash

#PBS -N run_cls
#PBS -V
#PBS -q debug
#PBS -l select=2:system=polaris
#PBS -l walltime=00:60:00
#PBS -l filesystems=home:grand
#PBS -A SolarWindowsADSP
#PBS -m be
#PBS -e /lus/grand/projects/SolarWindowsADSP/alanyahya/run_cls_e.sh
#PBS -o /lus/grand/projects/SolarWindowsADSP/alanyahya/run_cls_o.sh

HF_HOME=/lus/grand/projects/SolarWindowsADSP/alanyahya/.cache/huggingface
export HF_HOME=$HF_HOME

export http_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export https_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export ftp_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"

export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export CRAY_ACCEL_TARGET=nvidia80

cd /lus/grand/projects/SolarWindowsADSP/alanyahya/working
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

. /etc/profile

module use /soft/modulefiles; module load conda; conda activate base
echo python3: $(which python3)
python /lus/grand/projects/SolarWindowsADSP/alanyahya/hflogin.py

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

deepspeed \
    --hostfile="${DS_HOSTFILE}" \
    /grand/projects/SolarWindowsADSP/alanyahya/run_cls.py \
        --model_name_or_path Flamenco43/MatBERT \
        --dataset_name Flamenco43/nano-classification_dataset-40k \
        --trust_remote_code True \
        --max_seq_length 512 \
        --do_train \
        --do_eval \
        --eval_strategy epoch \
        --save_strategy epoch \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --num_train_epochs 8 \
        --learning_rate 5e-5 \
        --overwrite_output_dir \
        --include_tokens_per_second True \
        --output_dir /lus/grand/projects/SolarWindowsADSP/alanyahya/models/run_cls \
        --overwrite_output_dir True \
        --push_to_hub \
        --hub_private_repo True \
        --metric_name accuracy \
        --text_column_name text \
        --label_column_name label \
        --ddp_backend nccl \
        --deepspeed /grand/SolarWindowsADSP/alanyahya/ds_z3.json \
        --bf16 True \
        --bf16_full_eval True \
        --run_name bert-base-cls \