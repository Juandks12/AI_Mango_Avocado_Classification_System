"""Descarga los modelos entrenados (y, opcionalmente, los datasets) desde el
Release de GitHub del proyecto.

Uso:
    python download_assets.py              # solo los 8 modelos -> models/
    python download_assets.py --datasets   # además los 4 datasets -> data/
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "Juandks12/AI_Mango_Avocado_Classification_System"
TAG = "v1.0"
BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}"

MODELS = [
    "yolo_mango.pt", "yolo_aguacate.pt",
    "faster_mango.pth", "faster_aguacate.pth",
    "ssd_mango.pth", "ssd_aguacate.pth",
    "effdet_mango.pth", "effdet_aguacate.pth",
]
DATASETS = [
    "mango.v1i.yolov8.zip", "aguacate.v1i.yolov8.zip",
    "mango.v1i.coco.zip", "aguacate.v1i.coco.zip",
]

ROOT = Path(__file__).parent.resolve()


def download(name: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ya existe  {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = f"{BASE_URL}/{name}"
    print(f"  bajando    {name} ...", end="", flush=True)
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  bajando    {name} {done * 100 // total:3d}%", end="", flush=True)
    tmp.replace(dest)
    print(f"\r  listo      {dest.relative_to(ROOT)}          ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", action="store_true", help="descargar también los datasets de entrenamiento")
    args = parser.parse_args()

    print("Modelos:")
    for name in MODELS:
        download(name, ROOT / "models" / name)

    if args.datasets:
        print("Datasets:")
        for name in DATASETS:
            zip_path = ROOT / "data" / name
            target = ROOT / "data" / name.removesuffix(".zip")
            if target.exists():
                print(f"  ya existe  {target.relative_to(ROOT)}")
                continue
            download(name, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(target)
            zip_path.unlink()
            print(f"  extraído   {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
