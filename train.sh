python train.py --arch yolov5 \
                --depth s \
                --dataset Online \
                --path "/home/mq/data_disk2T/hieu/hoang/data_set/train" \
                --val-path "/home/mq/data_disk2T/hieu/hoang/data_set/val" \
                --batch-size 8 \
                --input-h 416 \
                --input-w 416 \
                --lr 1.25e-3 \
                --label-smoothing \
                --val 1 
                