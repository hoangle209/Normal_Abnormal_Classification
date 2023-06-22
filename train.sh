python train.py --arch yolov5 \
                --depth s \
                --path "/home/mq/data_disk2T/hieu/hoang/data_set/train" \
                --val-path "/home/mq/data_disk2T/hieu/hoang/data_set/val" \
                --input-h 416 \
                --input-w 416 \
                --lr 1.25e-3 \
                --label-smoothing \
                --val 1 \
                --batch-size 8