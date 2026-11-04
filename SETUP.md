# Setup on the RCI cluster

Replace `<user>` with your cluster login and `<email>` with your address.

## 1. Accounts

1. Get an RCI account.
2. Request access at <https://huggingface.co/facebook/sam3> and wait for approval.
3. Create a **Read** token at <https://huggingface.co/settings/tokens>.

## 2. Edit paths and email

| file | set |
|---|---|
| `flatsam2/path_config.py` | `OUTPUT_DIRECTORY`, `VIS_DIRECTORY`, `COMPARISONS_DIRECTORY` → `/mnt/personal/<user>/flatsam2/{results,vis,comparisons}/` |
| `experiments/pipeline.yml` | `OUTPUT_DIRECTORY`, `COMPARISONS_DIRECTORY` → same paths |
| `slurm/run.sbatch`, `slurm/evaluate.sbatch`, `slurm/cmd.sbatch` | `#SBATCH --mail-user=<email>` |
| `slurm/evaluate.sbatch` | `MAIL_TO="${FLATSAM2_MAIL_TO:-<email>}"` |
| `slurm/submit.py` | `--mail-to` default → `<email>` |
| `flatsam2/tracker.py` line 98 | `self.segmenter = SAM3StreamingTracker(debug=self.cfg.candidate_dump)` → `self.segmenter = SAM3StreamingTracker()` |

## 3. Push the code

Recommend putting the repo at `~/git/flatsam2/` on the RCI. The commands below, `.envrc`
and the `Makefile` all assume that path.

Create a `Makefile` in the repo root on the workstation (gitignored; recipe lines start with
a tab):

```make
REMOTE ?= <user>@login3.rci.cvut.cz:git/flatsam2/
RSYNC = rsync -avz --delete --exclude=.venv --exclude=.idea --exclude=slurm/logs

.PHONY: sync sync_dry
sync:
	$(RSYNC) --filter=":- .gitignore" --exclude=.git ./ $(REMOTE)
	$(RSYNC) .git $(REMOTE)
sync_dry:
	$(MAKE) sync RSYNC="$(RSYNC) -n"
```

Then push with `make sync`. `make sync_dry` lists what would change without changing it.

## 4. Create `.envrc` on the RCI

Create `~/git/flatsam2/.envrc` with:

```bash
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load torchvision/0.22.1-foss-2024a-CUDA-12.6.0
module load Triton/3.5.0-foss-2024a-CUDA-12.6.0
source ~/git/flatsam2/venv/bin/activate
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/mnt/personal/<user>/hf
export TORCH_HOME=/mnt/personal/<user>/torch
```

## 5. Create the venv on the RCI

Install everything with `--no-deps`. torch, torchvision, numpy and scipy must come from the
modules, not from pip.

```bash
cd ~/git/flatsam2
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
python -m venv venv
source .envrc
pip install --no-deps -e .
pip install --no-deps "xformers==0.0.31" "opencv-python-headless<4.11" \
    pydantic pydantic_core annotated_types typing_inspection "typing_extensions>=4.14" \
    tqdm einops scikit-image imageio tifffile lazy_loader networkx timm \
    huggingface_hub safetensors fastapi uvicorn
pip install --no-deps git+https://github.com/serycjon/ptlflow
pip install --no-deps git+https://github.com/serycjon/sam3
pip install --no-deps git+https://github.com/davnords/LoMa
mkdir -p /mnt/personal/<user>/flatsam2/results
```

Check:

```bash
python -c "import torch, numpy; print(torch.__file__, numpy.__version__)"
```

torch must load from `/mnt/appl/...` and numpy must be `1.26.4`. If not, remove the venv
copies:

```bash
pip uninstall -y torch triton numpy scipy $(pip list 2>/dev/null | awk '/^nvidia-/{print $1}')
```

If an import fails with `ModuleNotFoundError: X`, run `pip install --no-deps X`.

## 6. Download the weights (login node)

```bash
source .envrc
mkdir -p $HF_HOME
read -s -p "HF token: " T && printf '%s' "$T" > $HF_HOME/token && unset T && chmod 600 $HF_HOME/token
python -c "from sam3.model_builder import download_ckpt_from_hf as d; print(d(version='sam3'))"
python -c "from loma import LoMa, LoMaR; LoMa(LoMaR()); print('ok')"
```

## 7. Test

```bash
python -c "import sys; sys.argv=['run.py']; import run; print('imports ok')"
python slurm/submit_cmd.py --gpus 1 --job-name cudatest -- \
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`slurm/logs/cudatest_<jobid>.out` must show `True NVIDIA A100...`.

## 8. Run

```bash
cd ~/git/flatsam2 && source .envrc
python slurm/submit.py pot --config configs/tracker/loma_samh_matchnaming.py \
    --sequences V22_5 V29_3 --jobs 2 --time 00:30:00 --dry-run
```

Check the output dir and email, then run it again without `--dry-run`.

- Status: `squeue -u $USER`
- Logs: `slurm/logs/`. Errors are in `*_slot1.log`.
- Results: `/mnt/personal/<user>/flatsam2/results/<config>/POT-210/`
