# NOTE: This file is the content of the Model_nb22e_CNN cell to be inserted
# into birdclef-2026-eos-9.ipynb. It is NOT a standalone runnable script.
# Indentation is 4-space (it lives inside `if 'Model_nb22e_CNN' in ...:`).
# Adapted from 22e_ns1_ensemble.py CNN block (lines 619-815), stripped of
# Perch+MLP dependencies. The full nb22e pipeline includes Perch and MLP;
# this CNN-only variant skips both since eos-9 already has Perch coverage
# via Model_22 / 51 / 74.

if 'Model_nb22e_CNN' in _ensemble_models:

    _file_name_submission = "subm_nb22e_CNN.csv"

    # ---- Setup ----------------------------------------------------------
    import os, glob, time, gc
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torchaudio
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    import timm

    # Determinism (matches nb22e v2 seeded baseline LB 0.930)
    import random as _random_n22
    _SEED_n22 = 42
    _random_n22.seed(_SEED_n22)
    np.random.seed(_SEED_n22)
    torch.manual_seed(_SEED_n22)

    _device_n22 = "cpu"  # submission scoring is CPU-only

    # ---- Constants -------------------------------------------------------
    PERCH_SR     = 32_000
    N_CLASSES_n22 = 234
    SC_FILE_SEC  = 60
    OUTPUT_SLOTS = 12  # 60 / 5 = 12 five-sec slots per soundscape

    CNN_BACKBONE = "tf_efficientnet_b0_ns"
    CNN_N_MELS   = 224
    CNN_N_FFT    = 4096
    CNN_HOP      = 1252
    CNN_WIN_SEC  = 20
    CNN_WIN_SMP  = CNN_WIN_SEC * PERCH_SR
    CNN_STRIDE   = 5
    CNN_N_WIN    = 9                    # (60 - 20) / 5 + 1
    CNN_FILES_PER_BATCH = 4
    CNN_IO_WORKERS      = 4

    # ---- Paths -----------------------------------------------------------
    BASE_DIR_n22 = (
        Path("/kaggle/input/competitions/birdclef-2026")
        if Path("/kaggle/input/competitions/birdclef-2026").exists()
        else Path("/kaggle/input/birdclef-2026")
    )
    sample_sub_n22 = pd.read_csv(BASE_DIR_n22 / "sample_submission.csv")
    PRIMARY_LABELS_n22 = [c for c in sample_sub_n22.columns if c != "row_id"]
    assert len(PRIMARY_LABELS_n22) == N_CLASSES_n22, \
        f"expected {N_CLASSES_n22} species cols, found {len(PRIMARY_LABELS_n22)}"

    test_dir_n22 = BASE_DIR_n22 / "test_soundscapes"
    test_files_n22 = sorted(test_dir_n22.glob("*.ogg")) if test_dir_n22.exists() else []
    if not test_files_n22:
        # Staging fallback: regular kernel run has no real test files (only readme.txt).
        # Use a handful of train soundscapes so the pipeline exercises end-to-end.
        print("[nb22e_CNN] test_soundscapes empty -- staging fallback")
        test_files_n22 = sorted((BASE_DIR_n22 / "train_soundscapes").glob("*.ogg"))[:16]
    print(f"[nb22e_CNN] {len(test_files_n22)} test files")

    # ---- Load 5 fold checkpoints -----------------------------------------
    # Strict slug-based filter: only match files under our exact dataset slug.
    # The looser "ns1 in path" filter was unsafe when other model cells mount
    # datasets that happen to have fold*_best.pth files in their tree.
    _NS1_SLUG_FRAG = "birdclef-2026-cnn-ns1-checkpoints"
    _ckpt_hits = sorted(
        glob.glob(f"/kaggle/input/**/{_NS1_SLUG_FRAG}/**/fold*_best.pth",
                  recursive=True)
    )
    _ckpts = {}
    for _p in _ckpt_hits:
        _nm  = Path(_p).name
        _idx = int(_nm.replace("fold", "").replace("_best.pth", ""))
        if _idx not in _ckpts:
            _ckpts[_idx] = _p
    print(f"[nb22e_CNN] found {len(_ckpts)} NS R1 fold checkpoints")
    for _i, _p in sorted(_ckpts.items()):
        print(f"  fold {_i}: {_p}")
    assert len(_ckpts) == 5, "Expected 5 NS R1 checkpoints"

    # ---- Mel transform (fp32, no autocast) -------------------------------
    _mel_n22 = torchaudio.transforms.MelSpectrogram(
        sample_rate=PERCH_SR, n_mels=CNN_N_MELS, n_fft=CNN_N_FFT,
        hop_length=CNN_HOP, f_min=0, f_max=16_000, power=2.0,
        norm="slaney", mel_scale="htk",
    ).to(_device_n22)
    _db_n22 = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0).to(_device_n22)

    def _wav_to_mel_n22(wav_t):
        if wav_t.device.type != _device_n22:
            wav_t = wav_t.to(_device_n22)
        with torch.cuda.amp.autocast(enabled=False):
            wav_t = wav_t.float()
            mel = _mel_n22(wav_t)
            mel = _db_n22(mel)
            mel = torch.nan_to_num(mel, nan=-80.0, posinf=0.0, neginf=-80.0)
            mlo = mel.amin(dim=(1, 2), keepdim=True)
            mhi = mel.amax(dim=(1, 2), keepdim=True)
            mel = (mel - mlo) / (mhi - mlo + 1e-6)
            mel = torch.nan_to_num(mel, nan=0.0, posinf=1.0, neginf=0.0)
        return mel.unsqueeze(1).repeat(1, 3, 1, 1)

    # ---- Model architecture (matches nb22 training) ----------------------
    class _SEDHead_n22(nn.Module):
        def __init__(self, in_features, n_classes):
            super().__init__()
            self.fc_att = nn.Linear(in_features, n_classes)
            self.fc_cla = nn.Linear(in_features, n_classes)
        def forward(self, x):
            x = x.transpose(1, 2)
            att      = torch.softmax(torch.tanh(self.fc_att(x)), dim=1)
            framelg  = self.fc_cla(x)
            cliplg   = (att * framelg).sum(dim=1)
            return cliplg

    class _BirdCNN_n22(nn.Module):
        def __init__(self, n_classes=N_CLASSES_n22, in_chans=3):
            super().__init__()
            self.backbone = timm.create_model(
                CNN_BACKBONE, pretrained=False, in_chans=in_chans,
                num_classes=0, global_pool="", features_only=False,
            )
            with torch.no_grad():
                feat = self.backbone.forward_features(
                    torch.zeros(1, in_chans, CNN_N_MELS, 512)
                )
                self.feat_dim = feat.shape[1]
            self.head = _SEDHead_n22(self.feat_dim, n_classes)
        def forward(self, x):
            feat = self.backbone.forward_features(x).mean(dim=2)
            return self.head(feat)

    # ---- Load all 5 folds into memory ------------------------------------
    _models_n22 = []
    for _i in sorted(_ckpts.keys()):
        _m = _BirdCNN_n22().to(_device_n22)
        _m.load_state_dict(torch.load(_ckpts[_i], map_location=_device_n22))
        _m.eval()
        _models_n22.append(_m)
    print(f"[nb22e_CNN] loaded {len(_models_n22)} folds (fp32, {_device_n22})")

    _cn_n22 = os.cpu_count() or 4
    torch.set_num_threads(_cn_n22)

    # ---- Audio loading + chunking helpers --------------------------------
    def _load60_n22(path):
        try:
            wav, sr = torchaudio.load(str(path))
            if sr != PERCH_SR:
                wav = torchaudio.functional.resample(wav, sr, PERCH_SR)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            y = wav.squeeze(0).numpy().astype(np.float32)
        except Exception:
            return np.zeros(60 * PERCH_SR, dtype=np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        target = 60 * PERCH_SR
        return np.pad(y, (0, max(0, target - len(y))))[:target]

    def _absmax_n22(y):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        m = float(np.max(np.abs(y)))
        return (y / m) if m > 1e-8 else y

    def _file_to_chunks_n22(wav):
        starts = [i * CNN_STRIDE for i in range(CNN_N_WIN)]
        chunks = np.stack([
            _absmax_n22(wav[s * PERCH_SR:(s + CNN_WIN_SEC) * PERCH_SR])
            for s in starts
        ]).astype(np.float32)
        return np.ascontiguousarray(chunks)

    def _infer_chunks_n22(chunks_np):
        chunk_t = torch.from_numpy(chunks_np).to(_device_n22)
        chunk_t = torch.nan_to_num(chunk_t, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.inference_mode():
            mel = _wav_to_mel_n22(chunk_t)
            preds = None
            for _m in _models_n22:
                cliplg = _m(mel)
                p = torch.sigmoid(cliplg).float()
                preds = p if preds is None else preds + p
            preds = preds / len(_models_n22)
        return preds.cpu().numpy()

    def _windows_to_slots_n22(win_preds):
        n_files = win_preds.shape[0]
        out = np.zeros((n_files, OUTPUT_SLOTS, N_CLASSES_n22), dtype=np.float32)
        cnt = np.zeros(OUTPUT_SLOTS, dtype=np.float32)
        starts = [i * CNN_STRIDE for i in range(CNN_N_WIN)]
        for wi, ws in enumerate(starts):
            lo = ws // CNN_STRIDE
            hi = (ws + CNN_WIN_SEC) // CNN_STRIDE
            for slot in range(lo, hi):
                out[:, slot, :] += win_preds[:, wi, :]
                cnt[slot] += 1
        out /= cnt[None, :, None]
        return out

    # ---- Pipelined inference loop ----------------------------------------
    print(f"[nb22e_CNN] inference on {len(test_files_n22)} files "
          f"(batch={CNN_FILES_PER_BATCH}) ...", flush=True)
    _t0_n22 = time.time()
    _io_pool_n22 = ThreadPoolExecutor(max_workers=CNN_IO_WORKERS)
    def _load_batch_n22(paths):
        return [_load60_n22(p) for p in paths]

    _future_n22 = _io_pool_n22.submit(
        _load_batch_n22, test_files_n22[:CNN_FILES_PER_BATCH]
    )

    cnn_slot_pred_n22 = np.zeros(
        (len(test_files_n22), OUTPUT_SLOTS, N_CLASSES_n22), dtype=np.float32
    )
    for _bs in range(0, len(test_files_n22), CNN_FILES_PER_BATCH):
        _bp  = test_files_n22[_bs:_bs + CNN_FILES_PER_BATCH]
        _ba  = _future_n22.result()
        _nxt = _bs + CNN_FILES_PER_BATCH
        if _nxt < len(test_files_n22):
            _future_n22 = _io_pool_n22.submit(
                _load_batch_n22, test_files_n22[_nxt:_nxt + CNN_FILES_PER_BATCH]
            )
        _cpf  = np.stack([_file_to_chunks_n22(w) for w in _ba])
        _n    = _cpf.shape[0]
        _flat = _cpf.reshape(_n * CNN_N_WIN, CNN_WIN_SMP)
        _fp   = _infer_chunks_n22(_flat)
        _wp   = _fp.reshape(_n, CNN_N_WIN, N_CLASSES_n22)
        _sp   = _windows_to_slots_n22(_wp)
        for _fi in range(_n):
            cnn_slot_pred_n22[_bs + _fi] = _sp[_fi]
        if (_bs + _n) % 200 < CNN_FILES_PER_BATCH or (_bs + _n) >= len(test_files_n22):
            print(f"  [nb22e_CNN] {_bs + _n}/{len(test_files_n22)} files  "
                  f"elapsed={time.time()-_t0_n22:.0f}s", flush=True)
    _io_pool_n22.shutdown(wait=False)
    print(f"[nb22e_CNN] inference total: {time.time()-_t0_n22:.0f}s")

    # ---- Build submission CSV (canonical row_id + species order) ---------
    # cnn_slot_pred_n22 shape: (n_files, 12 slots, 234 classes)
    # row_id format: "{stem}_{end_sec}" where end_sec in {5,10,...,60}
    _rows = []
    _values = np.zeros(
        (len(test_files_n22) * OUTPUT_SLOTS, N_CLASSES_n22), dtype=np.float32
    )
    for _fi, _path in enumerate(test_files_n22):
        _stem = _path.stem
        for _wi in range(OUTPUT_SLOTS):
            _end_sec = (_wi + 1) * 5
            _rows.append(f"{_stem}_{_end_sec}")
            _values[_fi * OUTPUT_SLOTS + _wi] = cnn_slot_pred_n22[_fi, _wi, :]

    pred_df_n22 = pd.DataFrame(_values, columns=PRIMARY_LABELS_n22)
    pred_df_n22["row_id"] = _rows
    pred_df_n22 = pred_df_n22[["row_id"] + PRIMARY_LABELS_n22]

    # ALWAYS reindex against sample_submission's row_ids -- previous version had
    # an else-branch that emitted raw train_soundscapes row_ids during staging
    # fallback, which then NaN'd out the blend in eos-9 (Model_74 had test row_ids
    # from waveform_cache, ours had train row_ids -> no index overlap -> empty
    # blend). Unconditional left-merge: in real submission our predictions cover
    # all sample_sub row_ids and survive; in staging fallback our train row_ids
    # are filtered out and the output is sample_sub-aligned 0.0s (blend OK).
    sub_n22 = sample_sub_n22[["row_id"]].merge(pred_df_n22, on="row_id", how="left")
    sub_n22 = sub_n22.fillna(0.0)
    _n_nonzero_rows = int((sub_n22[PRIMARY_LABELS_n22].sum(axis=1) > 0).sum())
    print(f"[nb22e_CNN] sample_sub merge: {sub_n22.shape}, "
          f"{_n_nonzero_rows}/{len(sub_n22)} rows have non-zero predictions")

    assert sub_n22.columns.tolist() == ["row_id"] + PRIMARY_LABELS_n22, \
        "column order mismatch in subm_nb22e_CNN.csv"
    sub_n22.to_csv(_file_name_submission, index=False)
    print(f"[nb22e_CNN] saved {_file_name_submission}: shape={sub_n22.shape}")

    # Cleanup
    del _models_n22, cnn_slot_pred_n22, _mel_n22, _db_n22
    gc.collect()
