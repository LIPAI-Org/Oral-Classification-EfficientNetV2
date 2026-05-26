"""
Baixa e extrai os datasets necessários para o Projeto 2.

Uso:
    python scripts/setup_datasets.py

Execute a partir da raiz do repositório.
"""

import sys
import zipfile
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm

DATASET1_URL = "https://zenodo.org/records/17693395/files/Original%20ROI%20images.zip?download=1"
DATASET2_URL = "https://data.mendeley.com/public-files/datasets/bbmmm4wgr8/files/960934c4-81c1-44ca-88ac-ef0be632235d/file_downloaded"

# Raiz do projeto é um nível acima deste script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def download(url, dest):
    try:
        r = requests.get(url, stream=True, timeout=60, allow_redirects=True)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=dest.name) as bar:
            for chunk in r.iter_content(8192):
                bar.update(f.write(chunk))
        return True
    except requests.exceptions.RequestException as e:
        print(f"Erro no download: {e}")
        dest.unlink(missing_ok=True)
        return False


def extract(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for m in tqdm(zf.namelist(), desc="Extraindo"):
                zf.extract(m, dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as tf:
            for m in tqdm(tf.getmembers(), desc="Extraindo"):
                tf.extract(m, dest)
    else:
        print(f"Formato não reconhecido: {archive.name}")
        sys.exit(1)
    archive.unlink()


if __name__ == "__main__":
    DATASETS_DIR.mkdir(exist_ok=True)

    # Dataset 1 - OralEpitheliumDB
    dest1 = DATASETS_DIR / "Original ROI images"
    zip1  = DATASETS_DIR / "Original_ROI_images.zip"

    if dest1.exists() and any(dest1.iterdir()):
        print("Dataset 1 já existe, pulando.")
    else:
        print("Baixando Dataset 1 (OralEpitheliumDB)...")
        if download(DATASET1_URL, zip1):
            extract(zip1, DATASETS_DIR)
        else:
            print(f"Baixe manualmente em:\n  {DATASET1_URL}\ne extraia em 'datasets/'")

    # Dataset 2 - NDB-UFES
    dest2 = DATASETS_DIR / "images"
    zip2  = DATASETS_DIR / "NDB-UFES_images.zip"

    if dest2.exists() and any(dest2.iterdir()):
        print("Dataset 2 já existe, pulando.")
    else:
        print("Baixando Dataset 2 (NDB-UFES)...")
        if download(DATASET2_URL, zip2):
            extract(zip2, dest2)
        else:
            print("O Mendeley pode exigir login via navegador.")
            print("Baixe manualmente em: https://data.mendeley.com/datasets/bbmmm4wgr8/4")
            print("e extraia em 'datasets/images/'")

    # Verificação
    print("\nVerificando estrutura...")
    checks = {
        DATASETS_DIR / "manifest_split_oralepitheliumdb.csv": "CSV Dataset 1",
        DATASETS_DIR / "manifest_split_multiclass_NDB-UFES.csv": "CSV Dataset 2",
        DATASETS_DIR / "Original ROI images": "Imagens Dataset 1",
        DATASETS_DIR / "images": "Imagens Dataset 2",
    }
    for path, label in checks.items():
        status = "OK" if path.exists() else "FALTANDO"
        print(f"  [{status}] {label}")
