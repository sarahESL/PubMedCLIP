# QCR with PubMedCLIP

1. Use ```lib/utils/run.sh``` for creating dictionaries, ```img2idx.json```, resized images e.g. ```images64x64.pkl``` and other input files.
2. Move data to ```./data/data_rad``` and ```./data/data_slake``` for RAD and SLAKE datasets, respectively.
3. Set the ```CLIP_PATH```, i.e. path of the fine-tuned PubMedCLIP in your config file of interest in ```configs```.
4. Run the experiments for original CLIP and PubMedCLIP via ```run_clipOrg_ae.sh``` and ```run_pubmedclip_ae.sh```, respectively.
