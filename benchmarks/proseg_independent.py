"""Proseg comparison against independently imaged ComSeg benchmark boundaries.

Requires locally downloaded, unmodified members of Zenodo 11237477. The accepted
membrane/nuclear masks are a geometric reference, not known molecular origins.
No reference cell boundary is passed to Proseg. Nuclear seeds come from the
published input in_nucleus column, with 0 denoting unassigned transcripts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anndata as ad
import astrolabe as ab
import numpy as np
import pandas as pd
from scipy import ndimage, sparse
import tifffile


def label_map(labels, truth, selected):
    pairs = pd.crosstab(labels[selected], truth[selected])
    result = {}
    for label, counts in pairs.iterrows():
        winner = counts.idxmax()
        if label > 0 and winner > 0 and counts.max() / counts.sum() >= .9:
            result[int(label)] = int(winner)
    # Reject multiple source nuclei mapping to one accepted reference cell.
    frequencies = pd.Series(list(result.values())).value_counts()
    return {key: value for key, value in result.items() if frequencies[value] == 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = {"baysor": args.source / "baysor_segmentation.csv",
             "watershed": args.source / "4_8.0_df_crop9000_7_2.csv",
             "reference": args.source / "5_cell_min10_min10_int16.tif"}
    source = pd.read_csv(files["baysor"])
    watershed = pd.read_csv(files["watershed"])
    np.testing.assert_array_equal(source[["x", "y", "gene"]], watershed[["x", "y", "gene"]])
    # Predeclared central window, with 300-pixel context outside the scoring box.
    selected = source.x.between(3500, 5500) & source.y.between(3500, 5500)
    frame = source.loc[selected].copy().reset_index(drop=True)
    frame["watershed"] = watershed.loc[selected, "cell_id"].to_numpy()
    mask = tifffile.imread(files["reference"]).squeeze()
    truth = mask[frame.y.to_numpy(int), frame.x.to_numpy(int)]
    nuclei = frame.in_nucleus.to_numpy()
    mapping = label_map(nuclei, truth, nuclei > 0)
    interior = set()
    for label, bounds in enumerate(ndimage.find_objects(mask), 1):
        if bounds and all(s.start >= 3800 and s.stop <= 5201 for s in bounds):
            interior.add(label)
    scoring_cells = interior.intersection(mapping.values())
    score = np.isin(truth, list(scoring_cells)) & (nuclei == 0)
    if not score.any():
        raise RuntimeError("No independently scored extranuclear transcripts")
    genes = pd.Index(sorted(frame.gene.unique()))
    ids = pd.Index(sorted(set(nuclei) - {0}))
    nuclear = nuclei > 0
    counts = sparse.csr_matrix((np.ones(nuclear.sum(), dtype=int),
        (ids.get_indexer(nuclei[nuclear]), genes.get_indexer(frame.gene[nuclear]))), shape=(len(ids), len(genes)))
    data = ad.AnnData(counts, obs=pd.DataFrame(index=ids.astype(str)), var=pd.DataFrame(index=genes))
    data.obsm["spatial"] = frame.loc[nuclear].groupby("in_nucleus")[["x", "y"]].mean().reindex(ids).to_numpy() * .108
    data.uns["transcripts"] = pd.DataFrame({
        "transcript_id": source.index[selected].to_numpy(), "feature_name": frame.gene,
        "cell_id": pd.Series(nuclei).astype(str).where(nuclear, None),
        "x_location": frame.x * .108, "y_location": frame.y * .108,
        "z_location": 0., "overlaps_nucleus": nuclear})
    def accuracy(prediction):
        correct = prediction[score] == truth[score]
        per_cell = [float((prediction[score & (truth == cell)] == cell).mean()) for cell in sorted(scoring_cells)]
        return {"correct": int(correct.sum()), "scored": int(score.sum()),
                "transcript_accuracy": float(correct.mean()), "mean_cell_accuracy": float(np.mean(per_cell)),
                "per_cell_accuracy": per_cell}
    baseline = {}
    for name, column in (("watershed", "watershed"), ("published_baysor", "cell")):
        values = frame[column].to_numpy()
        # Align arbitrary comparator labels using only original nuclear spots.
        matching = label_map(values, truth, nuclear)
        predicted = pd.Series(values).map(matching).fillna(0).to_numpy(int)
        if name == "published_baysor":
            predicted[frame.is_noise.to_numpy(bool)] = 0
        baseline[name] = accuracy(predicted)
    report = {"source": "https://zenodo.org/records/11237477", "license": "CC-BY-4.0",
        "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
        "pixel_size_um": .108, "crop_pixels": [3500, 5500], "interior_pixels": [3800, 5200],
        "input_transcripts": len(frame), "input_nuclei": len(ids), "reference_cells_scored": len(scoring_cells),
        "reference_positive_transcripts": int((truth > 0).sum()), "extranuclear_transcripts_scored": int(score.sum()),
        "label_mapping": "unique nuclear majority >=90%; nuclear molecules excluded from scoring",
        "limitations": ["Geometric agreement, not known pre-spillage molecular origin",
                        "Only accepted reference cells completely inside the interior are scored",
                        "Published comparators used the full crop; Proseg uses a smaller context window",
                        "2D published benchmark collapses z stacks 2, 3, 4; no seed option in Proseg 3.2"],
        "baseline": baseline, "runs": []}
    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    for repeat in range(args.repeats):
        for diffusion in (False, True):
            started = time.perf_counter()
            result = next(iter(ab.seg.run_proseg(data, executable=args.executable,
                output_dir=args.output / f"repeat-{repeat}-diffusion-{diffusion}", diffusion=diffusion).values()))
            transcripts = result.uns["transcripts"].to_pandas()
            source_ids = result.obs.original_cell_id.astype(int).map(mapping)
            predicted = transcripts.cell_id.map(source_ids).fillna(0).to_numpy(int)
            np.testing.assert_array_equal(transcripts.transcript_id, data.uns["transcripts"].transcript_id)
            row = {"repeat": repeat, "diffusion": diffusion, "seconds": time.perf_counter() - started,
                   **accuracy(predicted), "accounting": result.uns["reconstruction"]["accounting"]}
            report["runs"].append(row)
            save()
            print(json.dumps({key: value for key, value in row.items() if key != "per_cell_accuracy"}), flush=True)


if __name__ == "__main__":
    main()
