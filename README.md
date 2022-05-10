# HomeWork_Speech_Processing
1. Sinh viên sử dụng dữ liệu đã thu (theo nhóm), trích xuất đặc trưng MFCC (39 đặc trưng, gồm cả MFCC, delta, deltadelta) của từng khẩu lệnh / con số
2. Viết chương trình sử dụng DTW để nhận dạng khẩu lệnh đơn lẻ (mỗi từ/khẩu lệnh dùng khoảng 2-3 mẫu)
3. Viết chương trình sử dụng HMM (segmental K-means) để nhận dạng khẩu lệnh đơn lẻ (sử dụng HMM với Mixture of Gaussians, sử dụng toàn bộ bộ dữ liệu đã thu).
* 2 chương trình sử dụng DTW và HMM có tên là:
  - recognition_by_dtw(path_wav_file)
  - recognition_by_hmm(path_wav_file)
* Link demo:
