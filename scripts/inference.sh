GPUS=8

model_path=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/liuyexin/workspace/model_weights/OmniGen2
transformer_path=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/liuyexin/workspace/model_weights/train_output/omnigen2/opensubject_v2_20251106_171008/checkpoint-40480/transformer
model_name=OmniGen2_opensubject
exp_name=opensubject_v2
output_dir=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/liuyexin/workspace/datasets/opensubject_evaluation/eval_results/${model_name}/${exp_name}_test
test_data=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/liuyexin/workspace/datasets/Opensubject_benchmark_omnicontext_format
openai_key="1955916774340870205"

accelerate launch --num_processes=${GPUS} -m osbench.inference \
--model_path ${model_path} \
--transformer_path ${transformer_path} \
--model_name "OmniGen2" \
--test_data ${test_data} \
--result_dir ${output_dir} \
--num_inference_step 50 \
--height 720 \
--width 1280 \
--text_guidance_scale 5.0 \
--image_guidance_scale 2.0 \
--num_images_per_prompt 1 \
--dtype 'fp16' \
--disable_align_res

python -m osbench.test_osbench_score \
--test_data ${test_data}  \
--result_dir ${output_dir} \
--model_name "OmniGen2" \
--openai_url "https://aigc.sankuai.com/v1/openai/native/chat/completions" \
--openai_key ${openai_key} \
--max_workers 16

echo "Calculating statistics"
python -m osbench.calculate_statistics \
--save_path ${output_dir} \
--model_name "OmniGen2" \
--backbone gpt4dot1