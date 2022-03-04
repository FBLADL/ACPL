import os

# JCL vanila
# for j in range(3):
#     for i in [5, 10, 15, 20]:
#         print('deploy {}:{}'.format(i, j + 1))
#         os.system(
#             'python main_lincls_single.py --download-name fa_121 --pretrained checkpoint_0099.pth.tar --v2mlp --gpu 1 --data-path ../DivideMix_Chest_multilabel/chestxray/ --user yu --warmup-epochs 20 --imagenet --resize 512 --ratio {} --runtime {}'.format(
#                 i, j + 1))
#         os.system('mkdir -p log/jcl_vanila/{}_semi/{}st_run'.format(i, j+1))
#         os.system(
#             'mv net1_epoch* result.csv log/jcl_vanila/{}_semi/{}st_run'.format(i, j + 1))

# MT
# for j in range(3):
#     for i in [2, 5, 10, 15, 20]:
#         print('deploy {}:{}'.format(i, j + 1))
#         os.system(
#             'python main_lincls_single_mt.py --download-name v2mlp --pretrained checkpoint_0099.pth.tar --v2mlp --gpu 1 --data-path ../DivideMix_Chest_multilabel/chestxray/ --user yu --warmup-epochs 20 --imagenet --resize 512 --ratio {} --runtime {} --batch-size 8'.format(
#                 i, j + 1))
#         os.system('mkdir -p log/MT/{}_semi/{}st_run'.format(i, j+1))
#         os.system(
#             'mv net1_epoch* result.csv log/MT/{}_semi/{}st_run'.format(i, j + 1))


# JCL MT
# for j in range(3):
#     for i in [2, 5, 10, 15, 20]:
#         print('deploy {}:{}'.format(i, j + 1))
#         os.system(
#             'python main_lincls_single_mt.py --download-name fa_121 --pretrained checkpoint_0099.pth.tar --v2mlp --gpu 1 --data-path ../DivideMix_Chest_multilabel/chestxray/ --user yu --warmup-epochs 20 --imagenet --resize 512 --ratio {} --runtime {} --batch-size 8'.format(
#                 i, j + 1))
#         os.system('mkdir -p log/jcl_MT/{}_semi/{}st_run'.format(i, j+1))
#         os.system(
#             'mv net1_epoch* result.csv log/jcl_MT/{}_semi/{}st_run'.format(i, j + 1))

# for j in range(3):
#     for i in [2, 5, 10, 15, 20]:
#         print('deploy {}:{}'.format(i, j + 1))
#         os.system(
#             'python main_lincls_single_mt.py --download-name v2mlp --pretrained checkpoint_0099.pth.tar --v2mlp --gpu 1 --data-path ../DivideMix_Chest_multilabel/chestxray/ --user yu --warmup-epochs 20 --imagenet --resize 512 --ratio {} --runtime {} --batch-size 8'.format(
#                 i, j + 1))
#         os.system('mkdir -p log/MT/{}_semi/{}st_run'.format(i, j+1))
#         os.system(
#             'mv net1_epoch* result.csv log/MT/{}_semi/{}st_run'.format(i, j + 1))

# densenet 121 imagenet pre-train
# for i in [2, 5, 10, 15, 20]:
#     print('deploy {}:{}'.format(i, 1))
#     os.system(
#         'python main_lincls_single.py --gpu 1 --data-path ../DivideMix_Chest_multilabel/chestxray/ --user yu --warmup-epochs 20 --imagenet --resize 512 --ratio {} --runtime {}'.format(
#             i, 1))
#     os.system('mkdir -p log/imgnet/{}_semi/{}st_run'.format(i, 1))
#     os.system(
#         'mv net1_epoch* result.csv log/imgnet/{}_semi/{}st_run'.format(i,  1))

# JCL MT 20%
# for j in [4,5]:
#     os.system(
#         'python main_lincls_single_mt.py --download-name v2mlp --pretrained checkpoint_0099.pth.tar --v2mlp --gpu 1 --data-path ../DivideMix_Chest_multilabel/chestxray/ --user yu --warmup-epochs 20 --imagenet --resize 512 --ratio {} --runtime {} --batch-size 8'.format(
#             20, j + 1))
#     os.system('mkdir -p log/MT/{}_semi/{}st_run'.format(20, j+1))
#     os.system(
#         'mv net1_epoch* result.csv log/MT/{}_semi/{}st_run'.format(20, j + 1))

# DenseCL MT ELR 2%
for j in range(1):
    for i in [5,10,15,20]:
        print('deploy {}:{}'.format(i, j + 1))
        os.system(
            f'python finetune.py --pretrained ck_099.pth.tar --mlp --gpu 1 --data /home/fengbei/Documents/data/chestxray/ --task chestxray14 --resize 512 --beta 0.9 --elr_weight 3 --epochs 30 --batch-size 8 --label_ratio {i} --runtime {j+1} ')
        os.system(f'mkdir -p /mnt/hd/TMI/DCL_ELR_MT/{i}_semi/{j+1}st')
        os.system(
            f'mv ck* result.csv /mnt/hd/TMI/DCL_ELR_MT/{i}_semi/{j+1}st')
