kill -9 $(lsof -t /dev/nvidia*)
sleep 1s

# vit-comer-tiny
#sh dist_train.sh configs/ade20k/upernet_vit_comer_tiny_512_160k_ade20k.py 8 --seed 2023

# vit-comer-small
#bash dist_train.sh configs/ade20k/upernet_vit_comer_small_512_160k_ade20k.py 2 --seed 2023

# vit-comer-base
bash dist_train.sh configs/ade20k/upernet_deit_adapter_base_512_160k_ade20k.py 4 --seed 2023

bash dist_train.sh configs/ade20k/upernet_uniperceiver_adapter_large_512_160k_ade20k.py 4 --seed 2023

bash dist_train.sh configs/ade20k/mask2former_beit_adapter_large_896_80k_ade20k_ms.py 4 --seed 2023

bash dist_train.sh configs/ade20k/upernet_beit_adapter_large_640_160k_ade20k_ms.py 4 --seed 2023

bash dist_train.sh configs/ade20k/upernet_beit_large_512_160k_ade20k_ms.py 4 --seed 2023


bash dist_train.sh configs/ade20k/mask2former_uniperceiver_adapter_large_512_40k_ade20k_ss_new2.py 4 --seed 2023

bash dist_train.sh configs/ade20k/upernet_uniperceiver_adapter_large_512_160k_ade20k_ss.py 4 --seed 2023

bash dist_train.sh configs/COS/Controllable_LPMoE_COD_uniperceiver.py 4 --seed 2024

bash dist_train.sh configs/COS/Controllable_LPMoE_COD_Beit.py 4 --seed 2024

bash dist_train.sh configs/COS/Controllable_LPMoE_PS_uniperceiver.py 4 --seed 2024

bash dist_train.sh configs/COS/Controllable_LPMoE_PS_Beit.py 4 --seed 2024

bash dist_train.sh configs/COS/Controllable_LPMoE_GD_GDD_uniperceiver.py 4 --seed 2024

bash dist_train.sh configs/COS/Controllable_LPMoE_GD_GDD_Beit.py 4 --seed 2024

#cd /opt/data/private/Syg/COD/code/test/9.26
# cd /opt/data/private/Syg/BS/code/12.8/

bash dist_train.sh configs/COS/Controllable_LPMoE_GD_Trans10k_uniperceiver.py 4 --seed 2024

bash dist_train.sh configs/COS/Controllable_LPMoE_SOD_uniperceiver.py 4 --seed 2024

bash dist_train.sh configs/COS/Controllable_LPMoE_MIS_SC18_Beit.py 4 --seed 2024

bash dist_train.sh configs/COS/Controllable_LPMoE_MIS_GS_uniperceiver.py 4 --seed 2024

bash dist_train.sh configs/COS/Controllable_LPMoE_COD_Beit_Baseline.py 4 --seed 2024

#python3 get_flops.py /opt/data/private/Syg/BS/code/11.28/work_dirs/mask2former_uniperceiver_adapter_large_512_40k_ade20k_ss_new2/mask2former_uniperceiver_adapter_large_512_40k_ade20k_ss_new2.py --shape 512
#/opt/data/private/Syg/BS/code/11.28/work_dirs/mask2former_uniperceiver_adapter_large_512_40k_ade20k_ss_new2/mask2former_uniperceiver_adapter_large_512_40k_ade20k_ss_new2.py



#python3 get_flops.py /opt/data/private/Syg/BS/code/9.29/work_dirs/mask2former_uniperceiver_adapter_large_512_40k_ade20k_ms/mask2former_uniperceiver_adapter_large_512_40k_ade20k_ms.py --shape 512

#python3 get_flops123.py /opt/data/private/Syg/BS/code/11.11/work_dirs/upernet_uniperceiver_adapter_large_512_160k_ade20k_ss/upernet_uniperceiver_adapter_large_512_160k_ade20k_ss.py --shape 512

#python3 get_flops123.py /opt/data/private/Syg/BS/code/11.11/work_dirs/mask2former_uniperceiver_adapter_large_512_40k_ade20k_ss/mask2former_uniperceiver_adapter_large_512_40k_ade20k_ss.py --shape 512



