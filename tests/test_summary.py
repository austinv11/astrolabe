import os

import astrolabe as ab


def test_summary_shows_source_path_relative_to_cwd(small_mdata, tmp_path, monkeypatch):
    # Hosted Windows runners can put the checkout and temp root on different
    # drives. This test exercises a relative path on one drive deliberately.
    monkeypatch.chdir(tmp_path)
    abs_path = str(tmp_path / "run_outs")
    small_mdata.uns["spatial"] = {"lib1": {"source_path": abs_path, "images": {}}}

    info = ab.summary(small_mdata)._info

    shown = info["spatial"]["lib1"]["source_path"]
    assert shown == os.path.relpath(abs_path)
    assert not os.path.isabs(shown)


def test_summary_keeps_absolute_path_across_windows_drives(monkeypatch):
    import ntpath
    from astrolabe._summary import _relpath
    windows_relpath = ntpath.relpath
    monkeypatch.setattr("astrolabe._summary.os.path.relpath", lambda path: windows_relpath(path, "D:/checkout"))
    assert _relpath("C:/data/run_outs") == "C:/data/run_outs"


def test_summary_detects_spatial_only_neighbors_and_prefixed_cluster_key(small_mdata):
    ab.tl.spatial_neighbors(small_mdata)
    ab.tl.cluster(small_mdata, resolution=0.5, neighbors_key="spatial_neighbors", key_added="spatial_leiden")

    s = ab.summary(small_mdata)
    info = s._info

    assert info["has_neighbors_graph"] is True
    assert "spatial_neighbors" in info["neighbor_graph_keys"]
    assert dict(info["workflow_steps"])["Neighbors"] is True
    assert dict(info["workflow_steps"])["Clustered"] is True
    assert "spatial_leiden" in info["obs_groups"].get("Clustering", [])

    html = s._repr_html_()
    assert "spatial_neighbors" in html
    assert "spatial_leiden" in html


def test_summary_neighbor_graph_keys_excluded_from_stored_results(small_mdata):
    ab.tl.spatial_neighbors(small_mdata)

    info = ab.summary(small_mdata)._info
    assert "spatial_neighbors" not in info["uns_results"]


def test_summary_no_neighbors_graph_by_default(small_mdata):
    info = ab.summary(small_mdata)._info
    assert info["has_neighbors_graph"] is False
    assert info["neighbor_graph_keys"] == []
    assert dict(info["workflow_steps"])["Neighbors"] is False
