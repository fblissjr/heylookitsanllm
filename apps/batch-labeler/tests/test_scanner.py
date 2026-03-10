"""Tests for scanner module: file hashing and directory scanning."""

from batch_labeler.scanner import file_hash, scan_images


class TestFileHash:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.jpg"
        f.write_bytes(b"fake image data here")
        assert file_hash(str(f)) == file_hash(str(f))

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.jpg"
        f2 = tmp_path / "b.jpg"
        f1.write_bytes(b"content one")
        f2.write_bytes(b"content two")
        assert file_hash(str(f1)) != file_hash(str(f2))

    def test_same_content_same_hash(self, tmp_path):
        f1 = tmp_path / "a.jpg"
        f2 = tmp_path / "b.jpg"
        data = b"identical content"
        f1.write_bytes(data)
        f2.write_bytes(data)
        assert file_hash(str(f1)) == file_hash(str(f2))


class TestScanImages:
    def test_finds_images(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"img")
        (tmp_path / "b.png").write_bytes(b"img")
        (tmp_path / "readme.txt").write_bytes(b"txt")
        result = scan_images(str(tmp_path))
        names = [p.name for p in result]
        assert "a.jpg" in names
        assert "b.png" in names
        assert "readme.txt" not in names

    def test_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.jpg").write_bytes(b"img")
        (tmp_path / "top.jpg").write_bytes(b"img")
        result = scan_images(str(tmp_path), recursive=True)
        assert len(result) == 2

    def test_non_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.jpg").write_bytes(b"img")
        (tmp_path / "top.jpg").write_bytes(b"img")
        result = scan_images(str(tmp_path), recursive=False)
        assert len(result) == 1
        assert result[0].name == "top.jpg"

    def test_sorted_output(self, tmp_path):
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            (tmp_path / name).write_bytes(b"img")
        result = scan_images(str(tmp_path))
        names = [p.name for p in result]
        assert names == ["a.jpg", "b.jpg", "c.jpg"]

    def test_custom_extensions(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"img")
        (tmp_path / "b.webp").write_bytes(b"img")
        result = scan_images(str(tmp_path), extensions={'.webp'})
        assert len(result) == 1
        assert result[0].name == "b.webp"

    def test_missing_dir_raises(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            scan_images("/nonexistent/path")

    def test_empty_dir(self, tmp_path):
        result = scan_images(str(tmp_path))
        assert result == []
