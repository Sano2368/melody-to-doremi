from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4, UUID
import subprocess
import sys

import hashlib
import shutil
import librosa
import torch
import torchcrepe
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

app = FastAPI()
folder = Path(__file__).resolve().parent

preview_folder = folder / "previews"
preview_folder.mkdir(exist_ok=True)
cache_folder = folder / "vocal_cache"
cache_folder.mkdir(exist_ok=True)


@app.get("/")
def home():
    return FileResponse(folder / "index.html")


@app.get("/font.otf")
def font():
    return FileResponse(
        folder / "CP Font.otf",
        media_type="font/otf"
    )


@app.get("/preview/{preview_id}")
def preview(preview_id: str):
    try:
        valid_id = UUID(preview_id).hex
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="音声がありません"
        )

    path = preview_folder / f"{valid_id}.wav"

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="音声がありません"
        )

    return FileResponse(path, media_type="audio/wav")


@app.post("/analyze")
def analyze(
    file: UploadFile = File(...),
    start: float = Form(0, ge=0),
    duration: float = Form(5, ge=1, le=60),
    separated: bool = Form(False),
    use_crepe: bool = Form(False)
):
    preview_url = None

    try:
        file.file.seek(0)

        if separated:
            # 分離済みの歌声は、指定区間をそのまま読み込む
            print("分離を省略して解析しています…", flush=True)

            audio, sample_rate = librosa.load(
                file.file,
                sr=22050,
                mono=True,
                offset=start,
                duration=duration
            )

        else:
            # 指定した区間を読み込む
            clip, _ = librosa.load(
                file.file,
                sr=44100,
                mono=False,
                offset=start,
                duration=duration
            )

            if clip.size == 0:
                return {
                    "message": "音声がありません。開始位置を確認してください。",
                    "notes": [],
                    "pitch_points": []
                }

            # ファイル名ではなく、音声の中身から識別する
            fingerprint = hashlib.sha256()
            fingerprint.update(b"htdemucs-cpu-44100-v1")
            fingerprint.update(str(clip.shape).encode("utf-8"))
            fingerprint.update(
                np.ascontiguousarray(clip, dtype=np.float32).tobytes()
            )

            cache_key = fingerprint.hexdigest()
            cached_vocals = cache_folder / f"{cache_key}.wav"

            if cached_vocals.is_file():
                print(
                    "保存済みの歌声を再利用します。分離は省略します。",
                    flush=True
                )

            else:
                with TemporaryDirectory(prefix="melody-") as temp:
                    temp_folder = Path(temp)
                    input_path = temp_folder / "clip.wav"
                    output_folder = temp_folder / "separated"

                    sf.write(input_path, clip.T, 44100)

                    print(
                        "初回のため、歌声を分離しています…",
                        flush=True
                    )

                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-m", "demucs",
                            "--two-stems=vocals",
                            "-n", "htdemucs",
                            "-d", "cpu",
                            "--out", str(output_folder),
                            str(input_path)
                        ],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=1800
                    )

                    if completed.returncode != 0:
                        print(completed.stdout)
                        print(completed.stderr)

                        return {
                            "message": (
                                "歌声の分離に失敗しました。"
                                "黒い画面のエラーを確認してください。"
                            ),
                            "notes": [],
                            "pitch_points": []
                        }

                    vocals_path = (
                        output_folder
                        / "htdemucs"
                        / "clip"
                        / "vocals.wav"
                    )

                    if not vocals_path.is_file():
                        return {
                            "message": "分離した歌声が見つかりません。",
                            "notes": [],
                            "pitch_points": []
                        }

                    # コピー完了後に正式な名前にする
                    pending_path = (
                        cache_folder / f"{uuid4().hex}.tmp"
                    )

                    try:
                        shutil.copyfile(vocals_path, pending_path)
                        pending_path.replace(cached_vocals)
                    finally:
                        pending_path.unlink(missing_ok=True)

                    print(
                        "歌声を保存しました。次回は再利用できます。",
                        flush=True
                    )

            # 保存した区間なので、ここでは切り出さない
            audio, sample_rate = librosa.load(
                cached_vocals,
                sr=22050,
                mono=True
            )

            print("音程を解析しています…", flush=True)

        if audio.size == 0:
            return {
                "message": "音声がありません。開始位置を確認してください。",
                "notes": [],
                "pitch_points": []
            }

        # 解析する歌声を試聴用に保存
        preview_id = uuid4().hex
        preview_path = preview_folder / f"{preview_id}.wav"

        sf.write(preview_path, audio, sample_rate)
        preview_url = f"/preview/{preview_id}"

        hop_length = 512

        if use_crepe:
            print("torchcrepeで解析しています…", flush=True)

            samples = torch.from_numpy(
                np.ascontiguousarray(audio, dtype=np.float32)
            ).unsqueeze(0)

            with torch.inference_mode():
                pitch, periodicity = torchcrepe.predict(
                    samples,
                    sample_rate,
                    hop_length,
                    float(librosa.note_to_hz("C2")),
                    2000.0,
                    model="full",
                    decoder=torchcrepe.decode.viterbi,
                    return_periodicity=True,
                    batch_size=256,
                    device="cpu"
                )

                # 無音で音程を拾わないようにする
                periodicity = torchcrepe.threshold.Silence(
                    -60.0
                )(
                    periodicity,
                    samples,
                    sample_rate,
                    hop_length
                )

            f0 = pitch.squeeze(0).cpu().numpy().copy()
            probability = periodicity.squeeze(0).cpu().numpy()

            # 周期性が弱い場所は音程なしとして扱う
            voiced = (
                np.isfinite(f0)
                & (f0 > 0)
                & (probability >= 0.21)
            )
            f0[~voiced] = np.nan

        else:
            print("pYINで解析しています…", flush=True)

            f0, voiced, probability = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sample_rate,
                hop_length=hop_length
            )
        doremi = {
            "C": "ド",
            "D": "レ",
            "E": "ミ",
            "F": "ファ",
            "G": "ソ",
            "A": "ラ",
            "B": "シ"
        }

        # 周波数を音程に変換
        midi_values = np.full(len(f0), np.nan)
        valid = voiced & np.isfinite(f0) & (f0 > 0)

        midi_values[valid] = librosa.hz_to_midi(f0[valid])
        smoothed = midi_values.copy()

        # 一瞬の音程の飛びをならす
        for i in range(2, len(midi_values) - 2):
            window = midi_values[i - 2:i + 3]

            if np.all(np.isfinite(window)):
                smoothed[i] = np.median(window)
        # 半音の境目での細かい切り替わりを抑える
        pitches = []
        held_pitch = None

        # 通常の境目は0.5半音。少し余裕を持たせる
        change_threshold = 0.65

        for value in smoothed:
            if not np.isfinite(value):
                pitches.append(None)
                held_pitch = None
                continue

            if held_pitch is None:
                held_pitch = int(round(value))

            elif abs(value - held_pitch) >= change_threshold:
                held_pitch = int(round(value))

            pitches.append(held_pitch)
        # 同じ音が続く区間に分ける
        runs = []
        run_start = 0

        for i in range(1, len(pitches) + 1):
            if i == len(pitches) or pitches[i] != pitches[run_start]:
                runs.append(
                    (run_start, i, pitches[run_start])
                )
                run_start = i

        joined = pitches.copy()

        # 同じ音に挟まれた短い揺れ・途切れをつなぐ
        for i in range(1, len(runs) - 1):
            left = runs[i - 1]
            middle = runs[i]
            right = runs[i + 1]

            first, last, pitch = middle
            seconds = (last - first) * hop_length / sample_rate
            surrounding_pitch = left[2]

            left_length = (
                (left[1] - left[0]) * hop_length / sample_rate
            )
            right_length = (
                (right[1] - right[0]) * hop_length / sample_rate
            )

            stable_sides = (
                left_length >= 0.09
                and right_length >= 0.09
            )

            small_change = (
                surrounding_pitch is not None
                and (
                    pitch is None
                    or abs(pitch - surrounding_pitch) <= 1
                )
            )

            if (
                seconds <= 0.12
                and surrounding_pitch == right[2]
                and stable_sides
                and small_change
            ):
                for frame in range(first, last):
                    joined[frame] = surrounding_pitch

        smoothed = np.array([
            np.nan if pitch is None else float(pitch)
            for pitch in joined
        ])

        notes = []
        current_midi = None
        start_frame = 0

        # 音程が変わった場所で音符を区切る
        for i in range(len(f0) + 1):
            midi = None

            if i < len(f0) and np.isfinite(smoothed[i]):
                midi = int(round(smoothed[i]))

            if midi != current_midi:
                if current_midi is not None:
                    start_seconds = (
                        start_frame * hop_length / sample_rate
                    )
                    end_seconds = min(
                        i * hop_length / sample_rate,
                        len(audio) / sample_rate
                    )
                    length = end_seconds - start_seconds

                    if length >= 0.05:
                        name = librosa.midi_to_note(current_midi)
                        label = doremi[name[0]] + name[1:]

                        notes.append({
                            "note": label,
                            "midi": current_midi,
                            "start": round(start + start_seconds, 2),
                            "duration": round(length, 2)
                        })

                current_midi = midi
                start_frame = i

        # グラフ用データ
        pitch_points = []

        for i in range(0, len(f0), 2):
            original = None
            corrected = None

            if voiced[i] and np.isfinite(f0[i]) and f0[i] > 0:
                original = round(
                    float(librosa.hz_to_midi(f0[i])), 2
                )

            if np.isfinite(smoothed[i]):
                corrected = int(round(smoothed[i]))

            pitch_points.append({
                "time": round(
                    start + i * hop_length / sample_rate, 3
                ),
                "original": original,
                "corrected": corrected
            })

        lines = []

        for note in notes:
            lines.append(
                f'{note["note"]}：'
                f'{note["start"]}秒から、'
                f'{note["duration"]}秒間'
            )

        if not notes:
            message = (
                "音程を検出できませんでした。"
                "歌声は試聴できます。"
            )
        else:
            message = (
                "分離済みの歌声を解析しました！\n"
                if separated
                else "歌声を分離して解析しました！\n"
            )
            message += "\n".join(lines)

        return {
            "message": message,
            "notes": notes,
            "pitch_points": pitch_points,
            "clip_start": start,
            "clip_duration": len(audio) / sample_rate,
            "preview_url": preview_url
        }

    except subprocess.TimeoutExpired:
        return {
            "message": (
                "分離に時間がかかりすぎました。"
                "短い区間で試してください。"
            ),
            "notes": [],
            "pitch_points": [],
            "preview_url": preview_url
        }

    except Exception as error:
        return {
            "message": f"解析エラー：{error}",
            "notes": [],
            "pitch_points": [],
            "preview_url": preview_url
        }