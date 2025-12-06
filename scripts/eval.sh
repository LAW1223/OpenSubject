# Change to project root directory
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/liuyexin/workspace/project/omnigen-mt-eval || exit 1

# Set PYTHONPATH to include current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Activate conda environment if needed
# Uncomment and set your environment name:
# source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/liuyexin/miniconda3/bin/activate
# conda activate omnigen2

# Or use conda Python directly (uncomment and set environment name):
# PYTHON_CMD=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/liuyexin/miniconda3/envs/<your_env_name>/bin/python
# If PYTHON_CMD is set, use it; otherwise use system python
PYTHON_CMD=${PYTHON_CMD:-python}

openai_key="1955916774340870205"

model_name=OmniGen2_opensubject
exp_name=opensubject_v1
output_dir_opensubject_v1=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/liuyexin/workspace/datasets/opensubject_evaluation/eval_results/${model_name}/${exp_name}_ours

${PYTHON_CMD} -m eval.omnicontext.test_opensubject_score \
--test_data /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/liuyexin/workspace/datasets/Opensubject_benchmark_omnicontext_format  \
--result_dir ${output_dir_opensubject_v1} \
--model_name "OmniGen2" \
--openai_url "https://aigc.sankuai.com/v1/openai/native/chat/completions" \
--openai_key ${openai_key} \
--max_workers 8

echo "Calculating statistics for ours..."
${PYTHON_CMD} -m eval.omnicontext.calculate_statistics \
--save_path ${output_dir_opensubject_v1} \
--model_name "OmniGen2" \
--backbone gpt4dot1