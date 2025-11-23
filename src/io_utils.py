import cv2
import ffmpeg

def video_reader(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps

def video_writer(frames_rgb, out_path, fps=30):
    h, w, _ = frames_rgb[0].shape
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{w}x{h}', r=fps)
        .output(out_path, vcodec='libx264', pix_fmt='yuv420p', r=fps, crf=18)
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )
    for f in frames_rgb:
        process.stdin.write(f.tobytes())
    process.stdin.close()
    process.wait()
#