# CRT-Fusion
./tools/dist_train.sh configs/crt-fusion/crtfusion-r50-fp16_phase1.py 4 --gpus 4 --work-dir /mnt/sda/radar_temporal/CRT-Fusion_phase1
python tools/swap_ema_and_non_ema.py /mnt/sda/radar_temporal/CRT-Fusion_phase1/iter_10548.pth
./tools/dist_train.sh configs/crt-fusion/crtfusion-r50-fp16_phase2.py 4 --gpus 4 --work-dir /mnt/sda/radar_temporal/CRT-Fusion_phase2 --resume-from /mnt/sda/radar_temporal/CRT-Fusion_phase1/iter_10548_ema.pth
python tools/swap_ema_and_non_ema.py /mnt/sda/radar_temporal/CRT-Fusion_phase2/iter_42192.pth
./tools/dist_test.sh configs/crt-fusion/crtfusion-r50-fp16_phase2.py /mnt/sda/radar_temporal/CRT-Fusion_phase2/iter_42192_ema.pth 1 --eval bbox


# CRT-Fusion-light
# ./tools/dist_train.sh configs/crt-fusion/crtfusion-r50-fp16_phase1_light.py 4 --gpus 4 --work-dir /mnt/sda/radar_temporal/CRT-Fusion_phase1_light
# python tools/swap_ema_and_non_ema.py /mnt/sda/radar_temporal/CRT-Fusion_phase1_light/iter_10548.pth
# ./tools/dist_train.sh configs/crt-fusion/crtfusion-r50-fp16_phase2_light.py 4 --gpus 4 --work-dir /mnt/sda/radar_temporal/CRT-Fusion_phase2_light --resume-from /mnt/sda/radar_temporal/CRT-Fusion_phase1_light/iter_10548_ema.pth
# python tools/swap_ema_and_non_ema.py /mnt/sda/radar_temporal/CRT-Fusion_phase2_light/iter_42192.pth
# ./tools/dist_test.sh configs/crt-fusion/crtfusion-r50-fp16_phase2_light.py /mnt/sda/radar_temporal/CRT-Fusion_phase2_light/iter_42192_ema.pth 1 --eval bbox


# 피클파일, 데이터 생성
# python tools/create_data_crtfusion.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
# python tools/generate_point_label.py
# python tools/radar_multi_sweeps.py