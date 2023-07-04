python train.py --arch yolov5 \
                --depth s \
                --dataset Online \
                --path "/home/mq/disk2T/hieu/new_data/train" \
                --val-path "/home/mq/disk2T/hieu/new_data/val" \
                --batch-size 8 \
                --input-h 416 \
                --input-w 416 \
                --lr 1.25e-3 \
                --val 1 \
                --loss focal

                