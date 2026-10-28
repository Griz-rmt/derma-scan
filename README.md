# Deteksi Penyakit Kulit (Streamlit)

## Deploy lewat GitHub + Streamlit Community Cloud

1. Taruh model di `models/model.pt` dan isi `models/labels.txt` (satu nama kelas per baris, sesuai urutan indeks keluaran model).
2. Push seluruh folder ini ke repository GitHub.
3. Buka [share.streamlit.io](https://share.streamlit.io), pilih repository, branch, dan main file `app.py`, lalu Deploy.

Catatan:
- Batas ukuran file GitHub adalah 100 MB. Jika `model.pt` lebih besar, pakai [Git LFS](https://git-lfs.com) atau unggah modelnya lewat sidebar aplikasi.
- `requirements.txt` memakai torch versi CPU supaya ukuran instalasi kecil.
- Jika repository publik, model ikut terlihat publik.

## Format model `.pt` yang didukung

| Format | Cara menyimpan |
|---|---|
| TorchScript (disarankan) | `torch.jit.script(model).save("model.pt")` atau `torch.jit.trace(model, contoh_input).save("model.pt")` |
| Model PyTorch utuh | `torch.save(model, "model.pt")` (kelas model harus bisa di-import) |
| YOLO (klasifikasi/deteksi) | Bobot Ultralytics, mis. `best.pt`. Tambahkan `ultralytics` dan `opencv-python-headless` ke `requirements.txt`, serta file `packages.txt` berisi `libgl1` |

`state_dict` saja tidak bisa dipakai karena tidak memuat arsitektur model.

Atur ukuran input dan normalisasi di sidebar agar sama dengan preprocessing saat training.

Hasil aplikasi ini bukan diagnosis medis.
