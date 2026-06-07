import pytest

from worldgap.data.loaders.hagrid import list_hagrid_sequences


def _make_fixture_hagrid_dir(tmp_path):
    for gesture in ["fist", "palm", "rock", "peace"]:  # only fist/palm are canonical
        gdir = tmp_path / gesture
        gdir.mkdir()
        (gdir / "seq_0001").mkdir()
            
    return tmp_path


def test_list_sequences_filters_to_canonical_gestures_only(tmp_path):
    root = _make_fixture_hagrid_dir(tmp_path)
    sequences = list_hagrid_sequences(root, gestures={"fist", "palm"})
    names = {p.parent.name for p in sequences}
    assert names == {"fist", "palm"}


def test_missing_directory_raises_with_actionable_message(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="download_datasets"):
        list_hagrid_sequences(missing)
