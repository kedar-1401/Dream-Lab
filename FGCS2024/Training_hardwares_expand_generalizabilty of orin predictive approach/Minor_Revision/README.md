The tokenized_finetune.pt files for BERT are same across all devices except 3090 
So, only the one checked in RPi5 is checked in to avoid issues with LFS.
When cloning or copying the folders, copy the .pt file from the below path except 3090
edge-ml-scheduler\Minor_Revision\Hardware_exp\Rpi\exp_scripts\BERT\new_tokenized\tokenized_finetune.pt
