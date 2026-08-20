"""RNAflux / fluxmap: de-novo subcellular domains from local RNA composition.

Adapted from Bento's ``flux`` / ``fluxmap`` (Mah et al. 2024, *Genome Biol*
25:82; https://github.com/YeoLab/bento-tools). Where
:mod:`astrolabe.tl._subcellular` asks *where in the cell* a gene sits against a
geometric reference, this asks the complementary question: **which recurring
molecular neighbourhoods does the cell interior decompose into?** Domains are
discovered from RNA *composition* rather than position, so they can pick out a
perinuclear ER-like zone or a secretory pole that no geometric partition names.

Two steps: :func:`flux` builds a per-pixel embedding of local composition, and
:func:`fluxmap` quantizes it into domains and dissolves them into per-cell
polygons.

Making this work on a targeted panel
------------------------------------
The obstacle is arithmetic. A composition vector needs enough transcripts in a
*subcellular* neighbourhood to be a composition rather than a one-hot, and
Bento's published runs were MERFISH on large cultured cells with ~10,000 genes
and thousands of transcripts each. On a 377-plex FFPE panel at ~56 transcripts
per cell, Bento's default radius (mean cell radius / 3, about 1.5 um) captures
~5 transcripts — 1.3% of the panel. Decomposing that directly yields components
of sampling noise.

What that costs, measured. On a synthetic benchmark built to match the real
failure mode -- 260 cells, 60 transcripts each, 400 genes, and two domains that
differ by a **modest proportional shift over half the genes with no gene
exclusive to either** -- recovery of the planted partition (adjusted Rand index
against the truth, mean over three SOM seeds) comes out as:

===================  ===============  ==========  =============
composition space    per-group        recovery    seed-to-seed
                     evidence         ARI         ARI
===================  ===============  ==========  =============
400 genes            0.05             0.03        0.02
8 modules            2.4              0.14        0.47
8 modules, 600 tx    45               0.72        0.86
===================  ===============  ==========  =============

So in gene space the method recovers nothing at this density, and in module space
it recovers real but weakly-resolved structure. A permutation control (gene labels
shuffled within each cell) returns ARI 0.0003, so the 0.14 is signal rather than
noise -- but 0.14 is a *detection*, not a clean segmentation, and low-plex domain
maps should be read that way. The earlier benchmark in this module planted
disjoint marker sets instead, where a single transcript identifies the domain;
that flattered the sparse regime badly and is why these numbers are the ones
quoted.

Three things make it tractable, and they compose:

1. **Composition over gene *groups*, not genes** (``gene_groups=``). This is
   the lever that decides whether the method works: a dozen co-expression
   modules is a space a 20-transcript neighbourhood can estimate; four hundred
   genes is not. :func:`~astrolabe.tl.som_gene_modules`
   already writes a suitable ``var`` column, and its ``unassigned`` residual
   class is kept as one explicit category rather than folded in silently.
2. **A Gaussian kernel instead of a hard disc** (``sigma_um=``). Nearby
   transcripts count for more, distant ones taper off, and the reported
   sample size becomes the *effective* one,
   ``n_eff = (sum w)^2 / sum w^2`` — the quantity that actually governs the
   variance of a weighted composition estimate.
3. **Shrinkage toward the cell's own composition** (``prior_strength=``). Each
   pixel's composition is a Dirichlet posterior mean,
   ``p = (sum w * counts + a * p_cell) / (sum w + a)``, so a pixel with few
   transcripts shrinks to its cell mean and contributes ~0 deviation instead of
   contributing noise. Measured effect: without it, low-evidence pixels carry
   *larger* deviations than well-evidenced ones (ratio 1.35, i.e. the noisiest
   pixels shout loudest); at the default ``a = 5`` that ratio is 1.05. It buys a
   little reproducibility and costs nothing, but it is second-order next to
   lever 1 -- do not expect it to rescue an under-powered run.

A floor on ``n_eff`` remains (``min_neighborhood_transcripts``): below some
density nothing recovers the signal, and silently returning domains would be
worse than refusing. Whatever the density, **check the permutation control** —
shuffle gene labels within each cell and confirm the domains disappear.

Divergences from published Bento
--------------------------------
Documented deliberately, in the manner of :func:`~astrolabe.pp.banksy_matrix`:

* **Neighbourhoods are restricted to the pixel's own cell.** Bento does this too;
  worth stating because the subtracted baseline is that cell's composition, so a
  neighbourhood spilling into a neighbour would compare a cell against a mixture
  that includes someone else.
* **Kernel weighting and shrinkage** (levers 2 and 3 above) are additions, not
  in the original.
* **Composition over gene groups** (lever 1) is an addition.
* **A real 2-D SOM with Ward meta-clustering**, trained through this library's
  :func:`~astrolabe.tl._som._train_som` (shuffled multi-epoch presentation, PCA
  initialization, radius annealed to one node, learning rate annealed to zero).
  Bento uses a degenerate ``1 x k`` grid trained sequentially at minisom
  defaults — a vector quantizer with no map topology and an order effect.
* **``n_domains`` is explicit**, not a ``kneed`` elbow over ``range(2, 9)``;
  :func:`~astrolabe.pl.som_quality_plot` and
  :func:`~astrolabe.tl.som_stability` already exist for judging the choice.
* **Polygonization reuses** :func:`astrolabe._shapes.merge_polygons` on the pixel
  lattice rather than adding ``rasterio``.
* Bento normalizes with ``StandardScaler(with_mean=False)`` and reduces with
  ``TruncatedSVD``; both are kept, since centring the already-differenced
  composition would undo the subtraction.

The Gaussian kernel here is ``exp(-0.5 (d/sigma)^2)`` — the same convention as
:func:`astrolabe.tl._gradient._edge_weights`, and a factor ``sqrt(2)`` narrower
than the one in :mod:`astrolabe.pp._banksy` at equal bandwidth.
"""
from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd

from astrolabe._utils import _resolve_transcripts, from_kwarg, with_adata
from astrolabe.tl._som import (
    _batch_bmu,
    _default_sigma,
    _metacluster_codebook,
    _quality_metrics,
    _require_minisom,
    _train_som,
    _unnormalized_umatrix,
)
from astrolabe.tl._subcellular import _prepare_geometry, _raster_points

__all__ = ["flux", "fluxmap"]

log = logging.getLogger(__name__)

_GATE_PROBE_CELLS = 200
_CELL_CHUNK = 2000
_KERNEL_TRUNCATION = 3.0  # evaluate the Gaussian out to 3 sigma


def _resolve_gene_groups(adata, gene_groups, gene_names):
    """Map gene names onto group codes.

    Returns ``(codes, group_names)`` where *codes* indexes *group_names* for
    every entry of *gene_names*. ``gene_groups=None`` is the identity mapping
    (one group per gene), which is Bento's behaviour and only appropriate on a
    high-plex panel.
    """
    if gene_groups is None:
        names = list(map(str, gene_names))
        return np.arange(len(names)), names

    if isinstance(gene_groups, str):
        if gene_groups not in adata.var:
            raise KeyError(
                f"var['{gene_groups}'] not found. Run "
                "ab.tl.som_gene_modules() first, or pass an explicit mapping."
            )
        lookup = adata.var[gene_groups].astype(str).to_dict()
    else:
        lookup = {str(k): str(v) for k, v in dict(gene_groups).items()}

    labels = np.array([lookup.get(str(g), "unassigned") for g in gene_names])
    group_names = sorted(set(labels.tolist()))
    # keep 'unassigned' last so it never leads a legend
    if "unassigned" in group_names:
        group_names.remove("unassigned")
        group_names.append("unassigned")
    index = {g: i for i, g in enumerate(group_names)}
    return np.array([index[l] for l in labels], dtype=np.int64), group_names


def _density_gate(median_eff, n_groups, sigma, min_abs, min_per_group,
                  *, estimated=None):
    """Return the refusal message when a neighbourhood is too sparse, else None.

    Separated from :func:`flux` so the same test can run on a cheap sample of
    cells before the full rasterization and again on the exact statistic.
    """
    per_group = median_eff / max(n_groups, 1)
    if median_eff >= min_abs and per_group >= min_per_group:
        return None
    why = (
        f"only {median_eff:.1f} effective transcripts"
        if median_eff < min_abs else
        f"{per_group:.3f} effective transcripts per composition group "
        f"({median_eff:.1f} spread over {n_groups} groups)"
    )
    where = (f", estimated on a random sample of {estimated} cells"
             if estimated is not None else "")
    return (
        f"flux: the median neighbourhood holds {why} at sigma={sigma:.2f} um"
        f"{where}. That is too little evidence to estimate a composition -- the "
        "vector is closer to a one-hot than to a proportion, and the "
        "decomposition below would return components of sampling noise. Options, "
        "in order: pass gene_groups= (e.g. 'som_gene_module') to reduce the "
        "composition space, which is by far the most effective; raise sigma_um; "
        "or use ab.tl.subcellular_coords/subcellular_enrichment, which needs no "
        "local composition at all. Pass force=True to run anyway."
    )


@with_adata
def flux(
    adata: ad.AnnData,
    transcripts=None,
    *,
    x_col: str = "x_location",
    y_col: str = "y_location",
    feature_col: str = "feature_name",
    cell_id_col: str = "cell_id",
    cell_boundaries_key: str = "cell_boundaries",
    nucleus_boundaries_key: str = "nucleus_boundaries",
    gene_groups: str | dict | None = None,
    sigma_um: float | None = None,
    points_per_cell: int = 200,
    pixel_um: float | None = None,
    prior_strength: float = 5.0,
    n_components: int = 10,
    min_neighborhood_transcripts: float = 10.0,
    min_transcripts_per_group: float = 0.5,
    force: bool = False,
    seed: int = 0,
    key_added: str = "flux",
) -> pd.DataFrame:
    """Per-pixel local RNA composition embedding (Bento's RNAflux).

    See the module docstring for why a targeted panel needs ``gene_groups=``, and
    for what the kernel and the shrinkage prior are doing.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    transcripts
        Transcript table; ``None`` uses ``uns['transcripts']``.
    x_col, y_col, feature_col, cell_id_col
        Transcript table columns.
    cell_boundaries_key, nucleus_boundaries_key
        ``obsm`` polygon keys. Both are required: the cell polygon bounds the
        raster and the neighbourhood, the nucleus is carried for downstream
        interpretation.
    gene_groups
        How to reduce the composition space. A ``var`` column name (e.g.
        ``'som_gene_module'``), an explicit ``{gene: group}`` mapping, or
        ``None`` for one group per gene. **On a targeted panel this is not
        optional in practice** -- see the module docstring.
    sigma_um
        Gaussian kernel bandwidth in um. ``None`` uses one third of the mean cell
        radius, Bento's radius rule reinterpreted as a bandwidth. The kernel is
        truncated at 3 sigma.
    points_per_cell
        Target interior pixels per cell; the lattice step is
        ``sqrt(area / points_per_cell)``, so cost is linear in cells.
    pixel_um
        Fixed lattice step in um, overriding *points_per_cell*.
    prior_strength
        Dirichlet prior weight ``a`` pulling each pixel's composition toward its
        own cell's. In units of transcripts: ``a = 5`` means a pixel with an
        effective 5 transcripts is shrunk halfway to the cell mean. ``0``
        disables shrinkage and reproduces the unregularized estimate.
    n_components
        Truncated-SVD components, capped by the number of groups.
    min_neighborhood_transcripts
        Refuse to run if the median **effective** neighbourhood size is below
        this.
    min_transcripts_per_group
        Refuse to run if the median effective neighbourhood, spread over the
        composition groups, gives fewer than this many transcripts per group.
        This is the binding constraint in practice and the one that separates a
        composition from a one-hot: on the benchmark in the module docstring,
        gene space sits at 0.05 per group and recovers nothing, while module
        space at 2-4 per group recovers real structure.
    force
        Run anyway, with a warning.
    seed
        Random seed for the SVD.
    key_added
        Prefix for stored outputs.

    Stores
    ------
    ``uns[key_added]``
        ``group_names``, ``variance_ratio``, ``params`` (including the measured
        ``median_n_eff`` and the groups-to-transcripts ratio).

    Returns
    -------
    pd.DataFrame
        One row per pixel: ``cell_id``, ``x``, ``y``, ``n_eff``, and
        ``flux_0 .. flux_{k-1}``.
    """
    from scipy.spatial import cKDTree
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    import scipy.sparse as sp
    import shapely

    transcripts = _resolve_transcripts(adata, transcripts, "flux")
    try:
        import pyarrow as pa
        if isinstance(transcripts, pa.Table):
            transcripts = transcripts.to_pandas()
    except ImportError:
        pass

    tx = transcripts[[cell_id_col, x_col, y_col, feature_col]].copy()
    tx[cell_id_col] = tx[cell_id_col].astype(str)

    (cell_poly, _cell_bnd, _nuc_poly, _nuc_bnd, _nuc_cent, cell_area,
     usable) = _prepare_geometry(adata, cell_boundaries_key, nucleus_boundaries_key)

    obs_index = {cid: i for i, cid in enumerate(adata.obs_names)}
    ci_raw = tx[cell_id_col].map(obs_index).to_numpy(dtype=np.float64)
    known = np.isfinite(ci_raw)
    ci = np.where(known, ci_raw, 0).astype(np.int64)
    keep = known & usable[ci]
    tx = tx[keep].reset_index(drop=True)
    ci = ci[keep]
    if len(tx) == 0:
        raise ValueError(f"No transcripts matched obs_names via '{cell_id_col}'")

    gene_names = np.unique(tx[feature_col].astype(str).to_numpy())
    gene_of_tx = pd.Categorical(
        tx[feature_col].astype(str), categories=gene_names
    ).codes.astype(np.int64)
    grp_of_gene, group_names = _resolve_gene_groups(adata, gene_groups, gene_names)
    grp_of_tx = grp_of_gene[gene_of_tx]
    n_groups = len(group_names)

    pts = np.column_stack([tx[x_col].to_numpy(dtype=np.float64),
                           tx[y_col].to_numpy(dtype=np.float64)])
    tree = cKDTree(pts)

    mean_radius = float(np.mean(np.sqrt(cell_area[usable] / np.pi)))
    sigma = float(sigma_um) if sigma_um is not None else mean_radius / 3.0
    cutoff = _KERNEL_TRUNCATION * sigma

    cells = np.flatnonzero(np.bincount(ci, minlength=adata.n_obs) > 0)
    if pixel_um is not None:
        step = np.full(adata.n_obs, float(pixel_um))
    else:
        with np.errstate(invalid="ignore"):
            step = np.sqrt(np.maximum(cell_area, 0.0) / max(points_per_cell, 16))
        step[~np.isfinite(step) | (step <= 0)] = 0.5

    log.info(
        "flux: %d transcripts over %d gene group(s) (from %d genes) in %d cells; "
        "sigma=%.2f um (mean cell radius %.2f um), prior_strength=%.1f",
        len(tx), n_groups, len(gene_names), len(cells), sigma, mean_radius,
        prior_strength,
    )

    # ---- per-cell composition in GROUP space -----------------------------
    cell_counts = sp.csr_matrix(
        (np.ones(len(ci)), (ci, grp_of_tx)), shape=(adata.n_obs, n_groups)
    ).toarray()
    cell_tot = cell_counts.sum(axis=1)
    cell_comp = cell_counts / np.maximum(cell_tot, 1.0)[:, None]

    def _neighborhoods(batch):
        """Rasterize a batch of cells and kernel-weight each pixel's own-cell
        transcripts. Returns None if the batch yielded no usable pixel."""
        gx_l, gy_l, gc_l = [], [], []
        for c in batch:
            gxc, gyc = _raster_points(cell_poly[c], step[c])
            gx_l.append(gxc); gy_l.append(gyc)
            gc_l.append(np.full(len(gxc), c, dtype=np.int64))
        gx = np.concatenate(gx_l); gy = np.concatenate(gy_l)
        gc = np.concatenate(gc_l)
        inside = shapely.contains_xy(cell_poly[gc], gx, gy)
        gx, gy, gc = gx[inside], gy[inside], gc[inside]
        if len(gx) == 0:
            return None

        nbrs = tree.query_ball_point(np.column_stack([gx, gy]), cutoff)
        n_raw = np.fromiter((len(n) for n in nbrs), dtype=np.int64, count=len(nbrs))
        if n_raw.sum() == 0:
            return None
        flat = np.concatenate([np.asarray(n, dtype=np.int64) for n in nbrs])
        owner = np.repeat(np.arange(len(nbrs)), n_raw)

        # a pixel's neighbourhood is its OWN cell only: the baseline subtracted
        # later is that cell's composition, so letting a neighbour's transcripts
        # in would compare a cell against a mixture containing someone else
        same_cell = ci[flat] == gc[owner]
        flat, owner = flat[same_cell], owner[same_cell]

        d = np.hypot(pts[flat, 0] - gx[owner], pts[flat, 1] - gy[owner])
        w = np.exp(-0.5 * (d / sigma) ** 2)
        sum_w = np.bincount(owner, weights=w, minlength=len(nbrs))
        sum_w2 = np.bincount(owner, weights=w ** 2, minlength=len(nbrs))
        with np.errstate(divide="ignore", invalid="ignore"):
            n_eff = np.where(sum_w2 > 0, sum_w ** 2 / sum_w2, 0.0)
        return gx, gy, gc, flat, owner, w, sum_w, n_eff

    # Refusing is the common outcome on a targeted panel, so it must not cost a
    # whole-tissue rasterization first: test the density on a sample of cells.
    if not force and len(cells) > _GATE_PROBE_CELLS:
        sample = np.sort(np.random.default_rng(seed).choice(
            cells, _GATE_PROBE_CELLS, replace=False))
        probed = _neighborhoods(sample)
        if probed is not None:
            early = _density_gate(
                float(np.median(probed[-1])), n_groups, sigma,
                min_neighborhood_transcripts, min_transcripts_per_group,
                estimated=len(sample))
            if early is not None:
                raise ValueError(early)

    px_l, py_l, pc_l, dev_l, neff_l = [], [], [], [], []
    for start in range(0, len(cells), _CELL_CHUNK):
        got = _neighborhoods(cells[start:start + _CELL_CHUNK])
        if got is None:
            continue
        gx, gy, gc, flat, owner, w, sum_w, n_eff = got

        wcounts = np.zeros((len(gx), n_groups))
        np.add.at(wcounts, (owner, grp_of_tx[flat]), w)

        # Dirichlet posterior mean toward the cell composition: a pixel with
        # little evidence returns the cell mean, so its deviation is ~0 rather
        # than noise
        prior = cell_comp[gc] * prior_strength
        denom = (sum_w + prior_strength)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            comp = np.where(denom > 0, (wcounts + prior) / denom, cell_comp[gc])
        dev = comp - cell_comp[gc]

        px_l.append(gx); py_l.append(gy); pc_l.append(gc)
        dev_l.append(dev.astype(np.float32)); neff_l.append(n_eff)

    if not dev_l:
        raise ValueError("no interior pixels were produced; check the polygons")
    cflux = np.vstack(dev_l)
    px = np.concatenate(px_l); py = np.concatenate(py_l)
    pc = np.concatenate(pc_l); n_eff = np.concatenate(neff_l)

    median_eff = float(np.median(n_eff))
    per_group = median_eff / max(n_groups, 1)
    msg = _density_gate(median_eff, n_groups, sigma,
                        min_neighborhood_transcripts, min_transcripts_per_group)
    if msg is not None:
        if not force:
            raise ValueError(msg)
        log.warning(msg)

    scaled = StandardScaler(with_mean=False).fit_transform(cflux)
    k = int(min(n_components, n_groups - 1, scaled.shape[0] - 1))
    if k < 1:
        raise ValueError(
            f"need at least 2 composition groups to decompose, got {n_groups}"
        )
    svd = TruncatedSVD(n_components=k, algorithm="randomized", random_state=seed)
    emb = svd.fit_transform(scaled).astype(np.float32)

    out = pd.DataFrame({
        "cell_id": pd.Categorical(adata.obs_names.to_numpy()[pc]),
        "x": px.astype(np.float32),
        "y": py.astype(np.float32),
        "n_eff": n_eff.astype(np.float32),
    })
    for j in range(k):
        out[f"flux_{j}"] = emb[:, j]

    adata.uns[key_added] = {
        "group_names": list(group_names),
        "gene_names": list(map(str, gene_names)),
        "variance_ratio": svd.explained_variance_ratio_.astype(np.float32),
        "params": {
            "sigma_um": sigma, "mean_cell_radius_um": mean_radius,
            "gene_groups": gene_groups if isinstance(gene_groups, str) else (
                None if gene_groups is None else "mapping"
            ),
            "n_groups": int(n_groups), "n_genes": int(len(gene_names)),
            "prior_strength": float(prior_strength),
            "points_per_cell": int(points_per_cell),
            "pixel_um": (None if pixel_um is None else float(pixel_um)),
            "n_components": k,
            "median_n_eff": median_eff,
            "min_neighborhood_transcripts": float(min_neighborhood_transcripts),
            "min_transcripts_per_group": float(min_transcripts_per_group),
            "median_n_eff_per_group": per_group,
            "forced": bool(force), "seed": int(seed),
        },
    }
    log.info(
        "flux: %d pixels x %d components; median effective neighbourhood %.1f "
        "transcripts over %d group(s) (%.1f per group), SVD retained %.1f%% "
        "-> uns['%s']",
        len(out), k, median_eff, n_groups, median_eff / max(n_groups, 1),
        100 * float(svd.explained_variance_ratio_.sum()), key_added,
    )
    return out


@with_adata(mirror_uns=from_kwarg("key_added"))
def fluxmap(
    adata: ad.AnnData,
    flux_embedding: pd.DataFrame,
    *,
    n_domains: int = 4,
    x: int | None = None,
    y: int | None = None,
    n_epochs: int = 10,
    sigma: float | None = None,
    learning_rate: float = 0.5,
    max_train_pixels: int = 500_000,
    min_n_eff: float = 0.0,
    polygons: bool = True,
    seed: int = 0,
    key_added: str = "fluxmap",
) -> pd.DataFrame:
    """Quantize :func:`flux` embeddings into de-novo subcellular domains.

    A SOM is trained on the pooled per-pixel embeddings, its codebook is Ward
    meta-clustered into *n_domains*, and each pixel takes its BMU's metacluster.
    Because the pixels come from every cell at once, a domain is a *recurring*
    molecular neighbourhood rather than a per-cell partition -- which is what
    makes the labels comparable between cells.

    **Validate before believing.** Re-run with gene labels shuffled within each
    cell: the domains should disappear. If they do not, the embedding is
    describing sampling noise and the domain map is an artifact of the
    decomposition, not a property of the tissue.

    Requires ``minisom`` (``uv add 'astrolabe[som]'``).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    flux_embedding
        The DataFrame returned by :func:`flux`.
    n_domains
        Number of reported domains -- the Ward cut through the codebook.
    x, y
        SOM grid. ``None`` uses the ``5*sqrt(n)``-nodes heuristic capped at
        30x30; pixels are plentiful and a larger grid buys nothing here.
    n_epochs, sigma, learning_rate
        Training parameters, as in :func:`~astrolabe.tl.som`.
    max_train_pixels
        Seeded training subsample. Every pixel is still assigned afterwards.
    min_n_eff
        Label pixels below this effective neighbourhood size ``'background'``
        instead of assigning them a domain. ``0`` keeps every pixel, which is
        reasonable when :func:`flux` was run with shrinkage: a low-evidence pixel
        already sits near the origin of the embedding.
    polygons
        Dissolve each cell's per-domain pixels into polygons and store them.
    seed
        Random seed.
    key_added
        Prefix for stored outputs.

    Stores
    ------
    ``obsm[key_added + '_fractions']``
        Float32 ``(n_obs, n_domains)`` per-cell domain occupancy, by pixel area.
    ``obsm[key_added + '{i}_boundaries']``
        Awkward polygon array per domain, when *polygons* is ``True``.
    ``uns[key_added]``
        ``domain_names``, ``weights``, ``umatrix``, ``hit_counts``,
        ``node_domain``, ``domain_profiles`` (mean embedding per domain), and
        ``params`` including QE/TE.

    Returns
    -------
    pd.DataFrame
        *flux_embedding* with a ``domain`` column added.
    """
    MiniSom = _require_minisom()

    cols = [c for c in flux_embedding.columns if c.startswith("flux_")]
    if not cols:
        raise ValueError(
            "flux_embedding has no 'flux_*' columns -- pass the DataFrame "
            "returned by ab.tl.flux()"
        )
    data = flux_embedding[cols].to_numpy(dtype=np.float32)
    size_col = "n_eff" if "n_eff" in flux_embedding.columns else "n_neighbors"
    ok = flux_embedding[size_col].to_numpy() >= min_n_eff
    n_px = int(ok.sum())
    if n_px == 0:
        raise ValueError(f"no pixel has {size_col} >= {min_n_eff}")

    side = max(4, min(30, int(np.ceil(np.sqrt(5 * np.sqrt(n_px))))))
    sx = x if x is not None else side
    sy = y if y is not None else (x if x is not None else side)
    sigma_eff = sigma if sigma is not None else _default_sigma(sx, sy)

    rng = np.random.default_rng(seed)
    train = data[ok]
    if len(train) > max_train_pixels:
        train = train[rng.choice(len(train), max_train_pixels, replace=False)]

    log.info(
        "fluxmap: training %dx%d grid on %d of %d pixels (%d dims, %d epochs) "
        "-> %d domains", sx, sy, len(train), n_px, data.shape[1], n_epochs, n_domains,
    )
    som_obj = _train_som(
        MiniSom, train, sx, sy, sigma=sigma_eff, learning_rate=learning_rate,
        n_epochs=n_epochs, seed=seed, init="pca",
    )
    bmu = _batch_bmu(som_obj, data)
    weights = som_obj.get_weights().astype(np.float32)
    node_domain, _ = _metacluster_codebook(weights, n_domains, bmu[ok])

    domain_of_px = np.full(len(data), -1, dtype=np.int32)
    domain_of_px[ok] = node_domain[bmu[ok]]
    present = sorted(set(domain_of_px[ok].tolist()))
    domain_names = [f"fluxmap{i + 1}" for i in range(len(present))]
    remap = {d: i for i, d in enumerate(present)}
    dom_idx = np.array([remap.get(d, -1) for d in domain_of_px], dtype=np.int32)

    qe, te = _quality_metrics(som_obj, train, seed)

    obs_index = {cid: i for i, cid in enumerate(adata.obs_names)}
    pc = flux_embedding["cell_id"].astype(str).map(obs_index).to_numpy(dtype=np.int64)
    n_dom = len(domain_names)
    counts = np.zeros((adata.n_obs, n_dom))
    good = dom_idx >= 0
    np.add.at(counts, (pc[good], dom_idx[good]), 1.0)
    tot = counts.sum(axis=1)
    fractions = np.full_like(counts, np.nan)
    fractions[tot > 0] = counts[tot > 0] / tot[tot > 0, None]

    profiles = np.zeros((n_dom, data.shape[1]), dtype=np.float64)
    for i in range(n_dom):
        m = dom_idx == i
        if m.any():
            profiles[i] = data[m].mean(axis=0)

    adata.obsm[key_added + "_fractions"] = fractions.astype(np.float32)
    adata.uns[key_added] = {
        "domain_names": domain_names,
        "weights": weights,
        "umatrix": _unnormalized_umatrix(weights),
        "hit_counts": np.bincount(bmu[ok], minlength=sx * sy).reshape(sx, sy),
        "node_domain": node_domain,
        "domain_profiles": profiles.astype(np.float32),
        "params": {
            "x": sx, "y": sy, "n_domains": int(n_domains),
            "n_domains_found": n_dom, "n_epochs": n_epochs,
            "sigma": sigma_eff, "learning_rate": learning_rate,
            "min_n_eff": float(min_n_eff), "seed": int(seed),
            "quantization_error": qe, "topographic_error": te,
        },
    }

    if polygons:
        _store_domain_polygons(adata, flux_embedding, pc, dom_idx, domain_names,
                               key_added)

    out = flux_embedding.copy()
    out["domain"] = pd.Categorical(
        np.where(dom_idx >= 0, np.asarray(domain_names + ["background"])[dom_idx],
                 "background"),
        categories=domain_names + ["background"],
    )
    log.info(
        "fluxmap: %d domains over %d pixels (QE=%.3f, TE=%.3f) -> "
        "obsm['%s_fractions'], uns['%s']",
        n_dom, n_px, qe, te, key_added, key_added,
    )
    return out


def _store_domain_polygons(adata, emb, pc, dom_idx, domain_names, key_added):
    """Dissolve each cell's pixels of one domain into polygons.

    Unions the pixel squares with shapely -- exact for a lattice, rather than
    adding ``rasterio`` for a raster-to-vector pass. **Every** component of the
    union is kept, so ``obsm`` holds a list of rings per cell rather than one
    ring: a cell's pixels of one domain are normally several disconnected
    patches, and keeping only the largest (what
    :func:`astrolabe._shapes.merge_polygons` is documented to do) would drop
    most of the region the method found.
    """
    import awkward as ak
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    xs = emb["x"].to_numpy(dtype=np.float64)
    ys = emb["y"].to_numpy(dtype=np.float64)

    # lattice step per cell, from the closest pair of distinct x values
    order = np.lexsort((xs, pc))
    steps = np.full(adata.n_obs, np.nan)
    pc_s, xs_s = pc[order], xs[order]
    d = np.diff(xs_s)
    pos = (pc_s[:-1] == pc_s[1:]) & (d > 1e-9)
    if pos.any():
        for c in np.unique(pc_s[:-1][pos]):
            steps[c] = float(d[pos & (pc_s[:-1] == c)].min())
    fallback = float(np.nanmedian(steps)) if np.isfinite(steps).any() else 0.5
    steps[~np.isfinite(steps)] = fallback

    # group pixel rows by (cell, domain) once, instead of scanning every cell for
    # every domain
    good = dom_idx >= 0
    key = pc[good].astype(np.int64) * (len(domain_names) + 1) + dom_idx[good]
    idx_all = np.flatnonzero(good)
    srt = np.argsort(key, kind="stable")
    key_s, idx_s = key[srt], idx_all[srt]
    bounds = np.flatnonzero(np.diff(key_s)) + 1
    starts = np.concatenate([[0], bounds])
    stops = np.concatenate([bounds, [len(key_s)]])

    per_domain: list[list] = [
        [[] for _ in range(adata.n_obs)] for _ in domain_names
    ]
    for s, e in zip(starts, stops):
        k = key_s[s]
        c, dom = divmod(int(k), len(domain_names) + 1)
        rows = idx_s[s:e]
        h = steps[c] / 2.0
        squares = [
            Polygon([(xs[r] - h, ys[r] - h), (xs[r] + h, ys[r] - h),
                     (xs[r] + h, ys[r] + h), (xs[r] - h, ys[r] + h)])
            for r in rows
        ]
        try:
            merged = unary_union(squares)
            parts = (list(merged.geoms)
                     if merged.geom_type == "MultiPolygon" else [merged])
            per_domain[dom][c] = [
                np.asarray(g.exterior.coords, dtype=np.float32)
                for g in parts if not g.is_empty
            ]
        except Exception:  # degenerate union -> leave the cell empty
            pass

    for i, name in enumerate(domain_names):
        adata.obsm[f"{key_added}{i + 1}_boundaries"] = ak.Array(per_domain[i])
