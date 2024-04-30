CUDA_VISIBLE_DEVICES=0 python main.py 0 &
CUDA_VISIBLE_DEVICES=1 python main.py 1 &
CUDA_VISIBLE_DEVICES=2 python main.py 2 &
CUDA_VISIBLE_DEVICES=3 python main.py 3 &
CUDA_VISIBLE_DEVICES=4 python main.py 4 &
CUDA_VISIBLE_DEVICES=5 python main.py 5 &
CUDA_VISIBLE_DEVICES=6 python main.py 6 &
CUDA_VISIBLE_DEVICES=7 python main.py 7 &
wait
