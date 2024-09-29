GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Change ``pretrain_model_path`` to use a different pretrain. 
# (e.g. GroundingDINO pretrain, DINO pretrain, Swin Transformer pretrain.)
# If you don't want to use any pretrained model, just ignore this parameter.

python -m torch.distributed.launch  --nproc_per_node=${GPU_NUM} main.py \
        --output_dir /home/Open-GroundingDino-V2/output \
        -c /home/Open-GroundingDino-V2/config/cfg_odvg.py \
        --datasets /home/Open-GroundingDino-V2/config/datasets_mixed_odvg_train.json \
        --pretrain_model_path /home/Open-GroundingDino-V2/groundingdino_swint_ogc.pth \
        --options text_encoder_type=/home/Open-GroundingDino-V2/bert-base-uncased
 