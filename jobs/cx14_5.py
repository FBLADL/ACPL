import os

for i in [5]:
    for j in range(5):
        print("deploy {}:{}".format(i, j + 1))
        os.system(
            f"CUDA_VISIBLE_DEEVICES=0,1 python main.py --data /home/fengbei/Documents/data/chestxray/ --task cx14 --mlp --resize 512 --batch-size 8 --epochs 20  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --eval-interval 1000 --desc lp_single_{i}_{j}_noelr_k50 --num-workers 4 --eval-interval 100 --t-p 0.7 --reinit --label_ratio {i} --beta 0.7 --stage1 1.0 --stage2 0.0 --stage3 0.0 --runtime {j} --topk 50 --pl-epochs 5"
        )
