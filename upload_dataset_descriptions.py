from huggingface_hub import create_repo, login, upload_folder
import dotenv
from pathlib import Path
import csv

dotenv.load_dotenv(override=True)
token = dotenv.get_key(dotenv_path=".env", key_to_get="HF_API_KEY")
login(token=token)

# Create the repo first
create_repo(repo_id="Cybernaut101/rPlan_subset_100", repo_type="dataset", exist_ok=True)

# Then upload
DATA_PATH = r"E:\Datasets\rPlan\dataset\floorplan_dataset_100"
root = Path(DATA_PATH)

# Build a manifest that pairs PNG and JSON by basename
pairs_csv = root / "pairs.csv"
rows = []
for png in root.rglob("*.png"):
    json_fp = png.with_suffix(".json")
    if json_fp.exists():
        rows.append({
            "id": png.stem,
            "image": png.relative_to(root).as_posix(),
            "annotation": json_fp.relative_to(root).as_posix(),
        })

# Write the manifest
pairs_csv.parent.mkdir(parents=True, exist_ok=True)
with pairs_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "image", "annotation"])
    writer.writeheader()
    writer.writerows(rows)

# Upload folder (includes pairs.csv)
upload_folder(
    folder_path=DATA_PATH,
    repo_id="Cybernaut101/rPlan_subset_100",
    repo_type="dataset",
    commit_message="Add/Update dataset with image-annotation manifest (pairs.csv)"
)