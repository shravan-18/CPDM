# train

# run with CPU
python3 main.py --seed 1234 --config configs/Template-CPDM.yaml --train --sample_at_start --save_top --gpu_ids -1
# python3 main.py --seed 1234 --config configs/Template-CPDM.yaml --train --sample_at_start --save_top --gpu_ids -1 --resume_model /path/to/resume/model/ckpt.pth -resume_optim /path/to/resume/optimizer/ckpt.pth

# run with GPU
# python3 main.py --seed 1234 --config configs/Template-CPDM.yaml --train --sample_at_start --save_top --gpu_ids 0
# python3 main.py --seed 1234 --config configs/Template-CPDM.yaml --train --sample_at_start --save_top --gpu_ids 0 --resume_model /path/to/resume/model/ckpt.pth -resume_optim /path/to/resume/optimizer/ckpt.pth

# test

# run with CPU
# python3 main.py --config configs/Template-CPDM.yaml --sample_to_eval --gpu_ids -1 --resume_model /path/to/resume/model/ckpt.pth

# run with GPU
# python3 main.py --config configs/Template-CPDM.yaml --sample_to_eval --gpu_ids 0 --resume_model /path/to/resume/model/ckpt.pth