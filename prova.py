import torch
checkpoint = torch.load('pretrained/rtmpose_coco_download_model/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth')
print(checkpoint.keys())
a = 1


checkpoint_coco = torch.load('/home/hao/project_2023/mmpose/pretrained/rtmpose-s_8xb256-420e_coco-256x192_coco/epoch_80_coco.pth')
print(checkpoint_coco.keys())
a = 1


checkpoint_jrdb = torch.load('/home/hao/project_2023/mmpose/pretrained/rtmpose-s_8xb256-420e_coco-256x192_coco/epoch_80_coco.pth')
print(checkpoint_jrdb.keys())
a = 1

checkpoint_coco_m = torch.load('pretrained/rtmpose_coco_download_model/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth')
print(checkpoint_coco_m.keys())
a = 1

checkpoint_coco_body8_s = torch.load('pretrained/rtmpose_coco_download_model/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth')
print(checkpoint_coco_body8_s.keys())
a = 1