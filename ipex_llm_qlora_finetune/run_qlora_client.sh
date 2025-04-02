Run_bitsandbtyes_llama3-3b() {
    model=Qwen/Qwen2-1.5B
    python bnb_lora_xpu.py --model_name ${model} --quant_type nf4 --device xpu --lora_r 8 --lora_alpha 16 --max_seq_length 128 --max_steps 50 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 | tee qlora.log
}

main() {
    Run_bitsandbtyes_llama3-3b
}

main
