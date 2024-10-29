### RUN isaac sim for data generation
# ./python.sh standalone_examples/replicator/offline_pose_generation/offline_pose_generation_changed.py

### RUN training pipline
# python -m torch.distributed.launch --nproc_per_node=1 train.py --data ../output/datasets_640x480/ --object xreal_with_usd_mat --outf output_xreal_300_640x480_2/weights --batchsize 16 --epochs 300

### RUN debug for visulization 
# python debug.y --data ../output/ 

### RUN in inference folder with inference.py
# python inference.py --data ../output/output/ --object xreal_with_usd_mat --debug --weights ../train/output_xreal/weights/net_epoch_99.pth
# 
### RUN in train2 folder with inference.py
# python infernece.py --data ../output/output/ --showbelief




#Laptop webcam intrinsics :
# Camera Matrix:
#  [[706.80501905   0.         325.62849036]
#  [  0.         702.05601553 256.58252382]
#  [  0.           0.           1.        ]]
# Distortion Coefficients:
#  [[ 5.56515460e-02  5.27339064e-01  2.69647253e-03 -3.30708521e-03
#   -5.34168695e+00]]


#Realsense D435i camera intrinsics: 
# Camera Matrix:
#  [[589.20903349   0.         341.32358359]
#  [  0.         593.14627632 245.52390956]
#  [  0.           0.           1.        ]]
# Distortion Coefficients:
#  [[-0.03401107  0.78303146  0.00662044  0.01100737 -3.04554863]]
