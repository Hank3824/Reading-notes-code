# FG-BERT_pytorch

## Requirements
- Python 3.7
- PyTorch 1.12.0 (CUDA 11.3 build optional)
- RDKit
- pandas, matplotlib, hyperopt, scikit-learn

## Environment setup (conda)
```bash
conda create -n FG-BERT python==3.7 -y
conda activate FG-BERT
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install rdkit
# conda install -c openbabel openbabel
pip install pandas matplotlib hyperopt scikit-learn
```

## Pre-training quick start
- Core files: `utils.py`, `transformer.py`, `bert.py`, `pretrain.py`
- Minimal run:
```bash
python pretrain.py
```

### Full CLI (run_pretrain.py)
You can launch pre-training with explicit arguments. Choose the example that matches your shell:

PowerShell (Windows):
```powershell
python .\run_pretrain.py `
  --data_path data\chembl_select_3\chembl_select_3.txt `
  --output_dir .\checkpoints `
  --batch_size 32 `
  --learning_rate 1e-4 `
  --max_epochs 50 `
  --patience 20 `
  --device auto `
  --model_name chembl_bert
```

CMD (Windows):
```bat
python .\run_pretrain.py ^
  --data_path data\chembl_select_3\chembl_select_3.txt ^
  --output_dir .\checkpoints ^
  --batch_size 32 ^
  --learning_rate 1e-4 ^
  --max_epochs 50 ^
  --patience 20 ^
  --device auto ^
  --model_name chembl_bert
```

Bash (Linux/macOS):
```bash
python ./run_pretrain.py \
  --data_path data/chembl_select_3/chembl_select_3.txt \
  --output_dir ./checkpoints \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --max_epochs 50 \
  --patience 20 \
  --device auto \
  --model_name chembl_bert
```