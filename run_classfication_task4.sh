python Classification_Train_.py \
    --input_dim 512 \
    --mlp_hidden 1024 \
    --batch_size 16 \
    --model_name 'resnet50' \
    --num_workers 32 \
    --devices 1 \
    --epochs 50 \
    --clr_temperature 0.5 \
    --learning_rate 0.0003 \
    --log_every_n_steps 40 \
    --model_save_path '/checkpoint/classfication/task4/category2/resnet50' \
    --task 4 \
    --category 2 \
    # --freeze option
 