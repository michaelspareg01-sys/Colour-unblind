import pytest

from colour_unblind.cli import main
from colour_unblind.processor import FrameProcessor
from colour_unblind.video import process_video
from conftest import make_clip, needs_ffmpeg, probe


@needs_ffmpeg
def test_process_video_keeps_frames_size_and_audio(tmp_path):
    src = make_clip(tmp_path / "in.mp4")
    out = tmp_path / "out.mp4"
    n = process_video(src, out, FrameProcessor("deutan", "recolour"), progress=False)
    info = probe(out)
    assert n == 10
    assert info == {"frames": 10, "size": (160, 90), "audio": True}


@needs_ffmpeg
def test_video_without_audio_works(tmp_path):
    src = make_clip(tmp_path / "in.mp4", audio=False)
    out = tmp_path / "out.mp4"
    process_video(src, out, FrameProcessor("cat", "simulate"), progress=False)
    assert probe(out)["audio"] is False


@needs_ffmpeg
def test_odd_frame_size_is_trimmed_to_even_for_playability(tmp_path):
    src = make_clip(tmp_path / "in.mkv", size="161x91")
    out = tmp_path / "out.mp4"
    process_video(src, out, FrameProcessor("tritan", "compare"), progress=False)
    assert probe(out)["size"] == (160, 90)


def test_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        process_video(tmp_path / "nope.mp4", tmp_path / "out.mp4", FrameProcessor("cat", "simulate"))


def test_refuses_to_overwrite_input(tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"")
    with pytest.raises(ValueError):
        process_video(src, src, FrameProcessor("cat", "simulate"))


@needs_ffmpeg
def test_cli_default_output_name(tmp_path, capsys):
    src = make_clip(tmp_path / "holiday.mp4")
    assert main(["compare", str(src), "--viewer", "cat", "--quiet"]) == 0
    assert probe(tmp_path / "holiday_compare_cat.mp4")["frames"] == 10


def test_help_lists_modes_and_viewers(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for word in ("simulate", "recolour", "compare", "live"):
        assert word in text
    with pytest.raises(SystemExit):
        main(["simulate", "--help"])
    text = capsys.readouterr().out
    for word in ("protan", "deutan", "tritan", "cat", "achromat"):
        assert word in text


def test_unknown_viewer_exits_nonzero(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["simulate", str(tmp_path / "x.mp4"), "--viewer", "dog"])
    assert exc.value.code != 0


def test_missing_file_is_a_clean_error(tmp_path, capsys):
    assert main(["simulate", str(tmp_path / "x.mp4")]) == 1
    assert "error" in capsys.readouterr().err


@needs_ffmpeg
def test_falls_back_to_opencv_without_ffmpeg(tmp_path, monkeypatch, capsys):
    src = make_clip(tmp_path / "in.mp4")
    monkeypatch.setattr("colour_unblind.video.shutil.which", lambda name: None)
    out = tmp_path / "out.mp4"
    assert process_video(src, out, FrameProcessor("protan", "simulate"), progress=False) == 10
    assert "without audio" in capsys.readouterr().err
    info = probe(out)
    assert info["frames"] == 10 and info["audio"] is False
