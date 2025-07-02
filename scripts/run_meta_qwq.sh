dataset_pair=(
    'gaia dev'
    # 'webwalker test_sample200'
    # 'gaia test_mm'
    # 'hle test'
    # 'browsecomp-zh test'
    # 'simpleqa test_sample200'
    #'gaia with_file'
)


python final_scripts/run_meta.py \
    --dataset_name $dataset_name \
    --split $split \
    --save_note "test_exp" \
    --use_single_dir \
    --save_dir "./results/" \
    --concurrent_limit 32 \
    --max_search_limit 15 \
    --bing_subscription_key "YOUR-BING-SUBSCRIPTION-KEY" \
    --api_base_url "http://0.0.0.0:8000/v1" \
    --model_name "QwQ-32B" \
    --aux_api_base_url "http://0.0.0.0:8001/v1" \
    --aux_model_name "Qwen2.5-32B-Instruct" \
    --omni_api_key 'YOUR-OMNI-API-KEY' 
