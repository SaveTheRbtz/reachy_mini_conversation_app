import soxr
import numpy as np
import pytest

from reachy_mini_conversation_app.realtime import (
    StreamingAudioBridge,
)


@pytest.mark.parametrize("output_rate", [16_000, 48_000])
@pytest.mark.parametrize("chunk_samples", [137, 2400, 9600])
def test_playback_resampling_matches_continuous_audio(output_rate: int, chunk_samples: int) -> None:
    """Preserve waveform continuity and duration regardless of incoming chunk boundaries."""
    bridge = StreamingAudioBridge(output_sample_rate=output_rate)
    pcm16 = (np.sin(np.arange(24_000) * 2 * np.pi * 440 / 24_000) * 16_000).astype("<i2")
    pcm16 = np.concatenate((pcm16, np.zeros(4096, dtype="<i2")))
    chunks = [
        bridge.pcm16_to_playback(pcm16[offset : offset + chunk_samples].tobytes())
        for offset in range(0, pcm16.size, chunk_samples)
    ]
    playback = np.concatenate(chunks)
    expected = soxr.resample(pcm16.astype(np.float32) / 32768.0, 24_000, output_rate)

    assert playback.dtype == np.float32
    assert output_rate < playback.size <= expected.size
    np.testing.assert_allclose(playback, expected[: playback.size], atol=1e-6)


def test_microphone_resampling_stays_continuous_through_silence() -> None:
    """Preserve input phase across live frames while silence carries the buffered speech tail."""
    bridge = StreamingAudioBridge(output_sample_rate=16_000)
    signal = np.sin(np.arange(16_000) * 2 * np.pi * 440 / 16_000).astype(np.float32) / 2
    microphone = np.concatenate((signal, np.zeros(4096, dtype=np.float32)))
    stereo = np.column_stack((microphone, microphone))
    pcm16 = b"".join(
        bridge.microphone_to_pcm16(16_000, stereo[offset : offset + 1024])
        for offset in range(0, stereo.shape[0], 1024)
    )
    actual = np.frombuffer(pcm16, dtype="<i2")
    expected = np.asarray(np.clip(soxr.resample(microphone, 16_000, 24_000), -1.0, 1.0) * 32767, dtype="<i2")

    assert 24_000 < actual.size <= expected.size
    np.testing.assert_allclose(actual, expected[: actual.size], atol=1)


@pytest.mark.parametrize("channel_first", [False, True], ids=["samples-first", "channels-first"])
@pytest.mark.parametrize("integer_pcm", [False, True], ids=["float32", "pcm16"])
def test_microphone_normalizes_and_downmixes_both_sdk_audio_layouts(channel_first: bool, integer_pcm: bool) -> None:
    """Preserve channel amplitude without changing the SDK-owned input frame."""
    channels = np.column_stack((np.full(2400, 0.25), np.full(2400, 0.75))).astype(np.float32)
    samples = (channels * 32768).astype(np.int16) if integer_pcm else channels
    if channel_first:
        samples = samples.T
    samples.setflags(write=False)
    original = samples.copy()
    bridge = StreamingAudioBridge(output_sample_rate=24_000)

    pcm = bridge.microphone_to_pcm16(24_000, samples)

    converted = np.frombuffer(pcm, dtype="<i2")
    assert converted.size == 2400
    np.testing.assert_allclose(converted, 0.5 * 32767, atol=1)
    np.testing.assert_array_equal(samples, original)
