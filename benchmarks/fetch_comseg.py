"""Explicitly fetch three CC BY 4.0 benchmark members using bounded ZIP ranges.

Source: Defard et al., https://doi.org/10.5281/zenodo.11237477 . This verifies
each extracted member's CRC and SHA-256, not the complete 3.5 GB archive's MD5.
"""
import argparse
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

URL = "https://zenodo.org/api/records/11237477/files/Ground_truth_benchmark_ComSeg.zip/content"
SIZE = 3473688307
PREFIX = "Ground_truth_benchmark_ComSeg/merfish_breast_cancer/"
MEMBERS = {
    "baysor_segmentation.csv": ("baysor/crop9000_7_2/segmentation.csv", "4d9b49c36f4c1091e552310a4f9acd31690073008a638da9c2179d8c6784966c"),
    "4_8.0_df_crop9000_7_2.csv": ("watershed/8.0_df_crop9000_7_2.csv", "30020ac5c732f66bc98f09d40aa6fadc9963eec2e7530ca3388ad5bd8d2b4a27"),
    "5_cell_min10_min10_int16.tif": ("ground_truth/croop9000_7_2/agreement_nuclei_cell_2503_local/cell_min10_min10_int16.tif", "2668f67d1097bbefab26064f25fb8ef0610564d304d2ba6025a50830177fdb5d"),
}


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.position = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else SIZE + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = SIZE - self.position if size < 0 else min(size, SIZE - self.position)
        if size > 100 * 1024**2:
            raise ValueError("Refusing a range above 100 MiB")
        if not size:
            return b""
        request = urllib.request.Request(URL, headers={"Range": f"bytes={self.position}-{self.position + size - 1}"})
        with urllib.request.urlopen(request, timeout=90) as response:
            if response.status != 206:
                raise RuntimeError("Server ignored bounded range request")
            data = response.read(size + 1)
        if len(data) != size:
            raise RuntimeError("Incorrect range length")
        self.position += size
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(RemoteZip()) as archive:
        for destination, (member, expected) in MEMBERS.items():
            content = archive.read(PREFIX + member)
            if hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError(f"Checksum mismatch: {member}")
            (args.output / destination).write_bytes(content)
            print(destination, len(content), flush=True)


if __name__ == "__main__":
    main()
