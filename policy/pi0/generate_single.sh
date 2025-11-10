data_dir=${1}
repo_id=${2}
uv run examples/aloha_real/convert_single_aloha_data_lerobot.py --raw_dir $data_dir --repo_id $repo_id
