# MEVF with PubMedCLIP

1. Follow the first and second step in README in `QCR_PUBMEDCLIP`.
2. Move data to ```data_RAD``` and ```data_SLAKE```, respectively.
3. Set the ```CLIP_PATH``` to the path of fine-tuned PubMedCLIP (with respect to your encoder of choice, i.e. RN50, RN50x4, ViT)
4. Run the experiments for original CLIP and PubMedCLIP via ```run_clipOrg.sh``` and ```run_pubmedclip.sh```, respectively.
