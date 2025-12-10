GPUS=4

# Configure paths - modify these according to your setup
model_path=/path/to/omnigen2_model
transformer_path=/path/to/opensubject_transformer
model_name=OmniGen2_opensubject
exp_name=opensubject_eval
output_dir=/path/to/results/${model_name}/${exp_name}
test_data=/path/to/osbench_dataset

# Set your OpenAI API key (or use environment variable)
openai_key="${OPENAI_API_KEY:-YOUR_API_KEY}"

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
--max_workers 8

echo "Calculating statistics"
python -m osbench.calculate_statistics \
--save_path ${output_dir} \
--model_name "OmniGen2" \
--backbone gpt4dot1