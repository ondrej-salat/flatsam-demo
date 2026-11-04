base_dir=/mnt/personal/serycjon/flatsam2/loma_tracker
for cfg_dir in $base_dir/*/; do
    cfg=$(basename "$cfg_dir")
    python evaluate.py pot --results-dir $base_dir --config /fake/${cfg}.py
    python evaluate.py pti --results-dir $base_dir --config /fake/${cfg}.py
done
