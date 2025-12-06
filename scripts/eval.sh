openai_key="1955916774340870205"

model_name=OmniGen2_opensubject
exp_name=opensubject_v1
output_dir=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/liuyexin/workspace/datasets/opensubject_evaluation/eval_results/${model_name}/${exp_name}_ours
test_data=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/liuyexin/workspace/datasets/Opensubject_benchmark_omnicontext_format

python -m osbench.test_osbench_score \
--test_data ${test_data}  \
--result_dir ${output_dir} \
--model_name "OmniGen2" \
--openai_url "https://aigc.sankuai.com/v1/openai/native/chat/completions" \
--openai_key ${openai_key} \
--max_workers 8

echo "Calculating statistics"
python -m osbench.calculate_statistics \
--save_path ${output_dir} \
--model_name "OmniGen2" \
--backbone gpt4dot1