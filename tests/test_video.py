import pytest

from jcc_agent.video import annotated_phase, extract_frames, iter_video_frames, probe_video


def test_annotation_boundaries_are_reference_only():
    assert annotated_phase(23.9) == "encounter"
    assert annotated_phase(24) == "transition"
    assert annotated_phase(31) == "pve"
    assert annotated_phase(133) == "augment"
    assert annotated_phase(149) == "later_gameplay"
    assert annotated_phase(-1) == "unannotated"


def test_extract_video_frames_and_reject_out_of_bounds(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = tmp_path / "测试视频.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for value in (0, 50, 100, 150, 200):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()
    info = probe_video(source)
    assert (info.width, info.height, info.frame_count) == (64, 48, 5)
    assert info.duration == .5
    frames = list(iter_video_frames(source, [.1, .3]))
    assert frames[0][0] == pytest.approx(.1)
    assert frames[1][1].mean() == pytest.approx(150, abs=5)
    output = extract_frames(source, tmp_path / "样本", [.1])
    assert output[0]["reference_source"] == "user_annotation_not_prediction"
    with pytest.raises(ValueError, match="outside"):
        list(iter_video_frames(source, [.5]))
