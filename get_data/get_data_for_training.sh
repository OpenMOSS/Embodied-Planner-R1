## 下载并处理得到 sft 和 task for rl


# 1. sft
ORIGIN_SFT='./sft/origin_sft'
mkdir ${ORIGIN_SFT}
git clone https://huggingface.co/datasets/agent-eto/eto-sft-trajectory ${ORIGIN_SFT}

python -m utils.modify_alf_sft --input ${ORIGIN_SFT}/data/alfworld_sft.json --output ./sft/alf.json
python -m utils.modify_alf_sft --input ${ORIGIN_SFT}/data/sciworld_sft.json --output ./sft/sci.json



# 2. task for rl
ALF_GAMEFILE_PATH='~/.cache/alfworld'
python -m utils.generate_alf_indice --input "${ALF_GAMEFILE_PATH}/json_2.1.1/train" --output ./rl/alf_train.json
python -m utils.generate_alf_indice --input "${ALF_GAMEFILE_PATH}/json_2.1.1/valid_seen" --output ./rl/alf_valid_seen.json
python -m utils.generate_alf_indice --input "${ALF_GAMEFILE_PATH}/json_2.1.1/valid_unseen" --output ./rl/alf_valid_unseen.json
python -m utils.generate_alf_indice --input "${ALF_GAMEFILE_PATH}/json_2.1.1/valid_train" --output ./rl/alf_valid_train.json

ORIGIN_INDICE='./eto_sci'
python -m utils.generate_sci_indice --input ${ORIGIN_INDICE}/train_indices.json --output ./rl/sci_train.json
python -m utils.generate_sci_indice --input ${ORIGIN_INDICE}/dev_indices.json --output ./rl/sci_dev.json
python -m utils.generate_sci_indice --input ${ORIGIN_INDICE}/test_indices.json --output ./rl/sci_test.json

