export PYTHONPATH=$PYTHONPATH:/mnt/ceph_rbd/len_rag


declare -A data=(
    ["musique"]="musique_500"
    ["bamboogle"]="test"
    ['nq']="test"
    ["popqa"]="test"
    ["triviaqa"]="test"
    ["hotpotqa"]="dev_500"
    ["2wikimultihopqa"]="2wiki_500"

)
declare -A model_paths=(
    ["qwen"]="${QWEN_MODEL_PATH}"
)


declare -A model_prompt_type=(
    ["qwen"]="none"   #pure_llama,qwen-qwq, deepseek, none 
)

prompt_types=("ours")  

for model_type in "${!model_paths[@]}"; do
    readarray -t paths <<< "${model_paths[$model_type]}"
    
    # 遍历该类型下的所有模型路径
    for model_path in "${paths[@]}"; do
        # 跳过空行
        [[ -z "$model_path" ]] && continue
        
        model_name=$(basename "$model_path")
        echo "Starting evaluation with model: $model_name (type: $model_type)"
        
        temp_config="/tmp/temp_config_${model_name}.yaml"
        cp "../src/eval_config.yaml" "$temp_config"
        
        sed -i "s|generator_model_path:.*|generator_model_path: \"${model_path//\//\\/}\"|" "$temp_config"
        sed -i "s|generator_model:.*|generator_model: \"${model_type}\"|" "$temp_config"
        
        echo "Current config for $model_name:"
        cat "$temp_config"

        for prompt_type in "${prompt_types[@]}"; do
            for dataset in "${!data[@]}"; do
                echo "Processing dataset: $dataset with model: $model_name"
                
                python -u ../../../evaluation/run_eval.py \
                    --config_path "$temp_config" \
                    --method_name lenrag \
                    --split "${data[$dataset]}" \
                    --dataset_name "$dataset" \
                    --prompt_type "$prompt_type" \
                    --prompt_template_type "${model_prompt_type[$model_type]}" \
                    --result_output_file "/mnt/ceph_rbd/TIRESRAG-R1/output/result_analysis/training/RL/result.json"
                
                if [ $? -ne 0 ]; then
                    echo "Error processing $dataset with model $model_name"
                    rm "$temp_config"
                    exit 1
                fi
            done
        done
        
        rm "$temp_config"
        echo "Completed evaluation with model: $model_name"
    done
done