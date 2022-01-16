# PubMedCLIP 
Source code for reproducing PubMedCLIP by fine-tuning CLIP using image--text pairs extracted from PubMed articles as provided in the ROCO dataset.
<br>
The fine-tuned PubMedCLIP models with ViT32, RN50 and RN50x4 visual encoders are also provided on [OneDrive](https://1drv.ms/u/s!ApXgPqe9kykTgn9wfsfX8JN0kjo-?e=dheCuW). In case of access permission issues, open an issue on this repository or contact <sedigheh.eslami@gmail.com>.
<br>
## How to use
This project requires Python 3.7 or newer.
1. Download ROCO dataset from [the official repository](https://github.com/razorx89/roco-dataset),
2. Create your virtual environment, activate it and install ``` pip install -r requirements.txt ```,
3. Set the input paths in `run.sh`,
4. Run `run.sh`.
## Citation
If you use this work in academic publication, please cite the [arXiv paper](https://arxiv.org/abs/2112.13906) by [Sedigheh Eslami](https://github.com/SarahESL), [Gerard de Melo](http://gerard.demelo.org/), and [Christoph Meinel](https://hpi.de/en/meinel/chair/prof-dr-ch-meinel.html):

```
Sedigheh Eslami, Gerard de Melo, Christoph Meinel (2021).
Does CLIP Benefit Visual Question Answering in the Medical Domain as Much as it Does in the General Domain?
arXiv e-prints 2112.13906, 2021.
```

BibTeX entry:
```
@article{EslamiDeMeloMeinel2021CLIPMedical,
  author = {{Eslami}, Sedigheh and {de Melo}, Gerard and {Meinel}, Christoph},
   title = {Does {CLIP} Benefit Visual Question Answering in the Medical Domain as Much as it Does in the General Domain?},
  journal = {arXiv e-prints},
  keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Artificial Intelligence, Computer Science - Computation and Language, Computer Science - Machine Learning},
  year = 2021,
  month = dec,
  eid = {arXiv:2112.13906},
  archivePrefix = {arXiv},
  eprint = {2112.13906},
  primaryClass = {cs.CV},
}
```
