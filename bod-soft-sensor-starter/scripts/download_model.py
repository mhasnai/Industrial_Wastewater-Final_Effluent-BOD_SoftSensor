#!/usr/bin/env python
"""
Placeholder downloader.
Replace the URL and checksum with your own hosting (Zenodo/OSF/Release asset).
"""
import pathlib, sys

def main():
    models = pathlib.Path("models")
    models.mkdir(exist_ok=True)
    target = models / "ext_model.joblib"
    target.write_bytes(b"")  # placeholder, harmless empty file
    print(f"Created placeholder at {target}. Replace with your actual weights.")

if __name__ == "__main__":
    main()