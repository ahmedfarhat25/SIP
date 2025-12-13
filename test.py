import librosa
import math
from typing import List
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def max_abs(x): #maximum absolute value in signal important at normalization step
    m = 0.0
    for v in x:
        av = v if v >= 0 else -v
        if av > m:
            m = av
    return m

def normalize(samples):# normalize audio signal to -1 to 1 range
    m = max_abs(samples)
    if m == 0:
        return samples[:]
    return [s / m for s in samples]

def hamming_window(N): #Hamming window function to reduce spectral leakage (make front and end of frame closer to zero)
    return [0.54 - 0.46 * math.cos((2 * math.pi * n) / (N - 1)) for n in range(N)]

def enframe(signal, frame_len, frame_step):# split signal into overlapping frames and zero-pad if necessary
    sig_len = len(signal)
    if sig_len <= frame_len:
        num_frames = 1
    else:
        num_frames = 1 + int(math.ceil((sig_len - frame_len) / frame_step))
    pad_len = int((num_frames - 1) * frame_step + frame_len)
    padded = signal[:] + [0.0] * (pad_len - sig_len)
    frames = []
    for i in range(num_frames):
        start = i * frame_step
        frames.append(padded[start:start+frame_len])
    return frames

def fft(x): #تحويل من time domain → frequency domain
    n = len(x)
    if n == 1:
        return [x[0]]
    if not is_power_of_two(n):
        m = next_pow_two(n)
        x = x + [0j]*(m-n)
        n = m
    return _fft_rec(x)

def _fft_rec(x): #recursive FFT implementation (divide and conquer(nlogn))
    n = len(x)
    if n == 1:
        return [x[0]]
    even = _fft_rec(x[0::2])
    odd  = _fft_rec(x[1::2])
    out = [0j]*n
    for k in range(n//2):
        ang = -2*math.pi*k/n
        wk = complex(math.cos(ang), math.sin(ang))
        t = wk * odd[k]
        out[k] = even[k] + t
        out[k+n//2] = even[k] - t
    return out

def is_power_of_two(n):
    return (n & (n - 1)) == 0 and n != 0

def next_pow_two(n):
    p = 1
    while p < n:
        p <<= 1
    return p

def magnitude_spectrum(frame, NFFT): #compute magnitude spectrum of a frame using FFT
    x = [complex(v,0) for v in frame] + [0j]*(NFFT-len(frame))
    X = fft(x)
    half = NFFT//2 + 1
    mags = [math.sqrt(X[k].real**2 + X[k].imag**2) for k in range(half)]
    return mags

def hz_to_mel(f):
    return 2595 * math.log10(1 + f/700)

def mel_to_hz(m):
    return 700*(10**(m/2595)-1)

def linspace(a,b,n):# generate n linearly spaced points between a and b
    if n == 1:
        return [a]
    step = (b-a)/(n-1)
    return [a+i*step for i in range(n)]

def mel_filterbank(num_filters, NFFT, sr):# create mel filterbank
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr/2)
    mel_points = linspace(low_mel, high_mel, num_filters+2)
    hz_points  = [mel_to_hz(m) for m in mel_points]
    bin_points = [int((NFFT+1)*hz/sr) for hz in hz_points]
    fbank = [[0.0]*(NFFT//2+1) for _ in range(num_filters)]
    for m in range(1, num_filters+1):
        left = bin_points[m-1]
        center = bin_points[m]
        right = bin_points[m+1]
        for k in range(left, center):
            fbank[m-1][k] = (k-left)/(center-left)
        for k in range(center, right):
            fbank[m-1][k] = (right-k)/(right-center)
    return fbank

def dct_type_2(mat, num_ceps):# compute DCT type II of each row in mat
    F = len(mat)
    N = len(mat[0])
    out = [[0.0]*num_ceps for _ in range(F)]
    for t in range(F):
        for k in range(num_ceps):
            s = 0.0
            for n in range(N):
                s += mat[t][n] * math.cos(math.pi*k*(2*n+1)/(2*N))
            out[t][k] = s
    return out

def compute_delta(feat, N=2):# compute delta (derivative) features from static features
    T = len(feat)
    if T == 0:
        return []
    D = len(feat[0])
    delta = [[0.0]*D for _ in range(T)]
    denom = 2 * sum(i*i for i in range(1,N+1))
    for t in range(T):
        for d in range(D):
            num = 0.0
            for n in range(1, N+1):
                t_minus = max(0, t-n)
                t_plus  = min(T-1, t+n)
                num += n*(feat[t_plus][d] - feat[t_minus][d])
            delta[t][d] = num/denom
    return delta

def compute_mfcc(signal, sr, n_mfcc=20, num_filters=26, # compute MFCC features from audio signal
                frame_size=0.025, frame_step=0.01,
                NFFT=512):
    frame_len = int(round(frame_size*sr))
    step_len  = int(round(frame_step*sr))
    frames = enframe(signal, frame_len, step_len)
    win = hamming_window(frame_len)
    mags = []
    for f in frames:
        wf = [f[i]*win[i] for i in range(frame_len)]
        mags.append(magnitude_spectrum(wf, NFFT))
    fbank = mel_filterbank(num_filters, NFFT, sr)
    energies = []
    for m in mags:
        e = []
        for fb in fbank:
            s = sum([m[k]*fb[k] for k in range(len(fb))])
            if s <= 0:
                s = 1e-10
            e.append(math.log(s))
        energies.append(e)
    mfcc_feat = dct_type_2(energies, n_mfcc)
    delta1 = compute_delta(mfcc_feat)
    delta2 = compute_delta(delta1)
    combined = [mfcc_feat[i]+delta1[i]+delta2[i] for i in range(len(mfcc_feat))]
    return combined

phonemes = {
    "aa": [0.1]*60,
    "ee": [0.5]*60,
    "oo": [1.0]*60
}

def euclidean_distance(vec1, vec2):
    return math.sqrt(sum((vec1[i]-vec2[i])**2 for i in range(len(vec1))))

def predict_phoneme(frame): # predict phoneme for a given MFCC frame using nearest neighbor
    best_phoneme = None
    min_dist = float("inf")
    for ph, proto in phonemes.items():
        dist = euclidean_distance(frame, proto)
        if dist < min_dist:
            min_dist = dist
            best_phoneme = ph
    return best_phoneme

# ==================== GUI APPLICATION ====================

class AcousticProcessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Acoustic Speech Processing - Manual Implementation")
        self.root.geometry("1200x800")
        
        self.audio_signal = None
        self.sample_rate = None
        self.mfcc_frames = None
        self.acoustic_sequence = None
        
        self.setup_gui()
    
    def setup_gui(self):
        # Menu Bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Audio", command=self.load_audio)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Process", menu=process_menu)
        process_menu.add_command(label="Extract MFCC", command=self.extract_mfcc)
        process_menu.add_command(label="Acoustic Recognition", command=self.acoustic_recognition)
        process_menu.add_command(label="Process All", command=self.process_all)
        
        # Main Frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Panel - Visualization
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.notebook = ttk.Notebook(left_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.waveform_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.waveform_frame, text="Waveform")
        
        self.mfcc_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.mfcc_frame, text="MFCC")
        
        # Right Panel - Console
        console_frame = ttk.LabelFrame(main_frame, text="Console Output", padding=10)
        console_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.console = scrolledtext.ScrolledText(console_frame, width=50, height=40,
                                                  font=("Consolas", 9))
        self.console.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Status Bar
        self.status_label = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.log("="*60)
        self.log("ACOUSTIC SPEECH PROCESSING - MANUAL IMPLEMENTATION")
        self.log("="*60)
        self.log("Load an audio file to begin...")
    
    def log(self, message):
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)
        self.root.update_idletasks()
    
    def load_audio(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                self.log(f"\n{'='*60}")
                self.log(f"Loading: {file_path}")
                
                y, sr = librosa.load(file_path, sr=None)
                self.audio_signal = y.tolist()
                self.sample_rate = sr
                
                self.log(f"Sample Rate: {sr} Hz")
                self.log(f"Duration: {len(self.audio_signal)/sr:.2f} seconds")
                self.log(f"Samples: {len(self.audio_signal)}")
                self.log(f"{'='*60}\n")
                
                self.plot_waveform()
                self.status_label.config(text=f"Audio loaded: {len(self.audio_signal)/sr:.2f}s")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load audio: {str(e)}")
                self.log(f"ERROR: {str(e)}")
    
    def plot_waveform(self):
        for widget in self.waveform_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(8, 4))
        time = [i/self.sample_rate for i in range(len(self.audio_signal))]
        ax.plot(time, self.audio_signal, linewidth=0.5)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform')
        ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=self.waveform_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def extract_mfcc(self):
        if self.audio_signal is None:
            messagebox.showwarning("Warning", "Load audio file first!")
            return
        
        try:
            self.log("\n" + "="*60)
            self.log("EXTRACTING MFCC FEATURES")
            self.log("="*60)
            
            self.mfcc_frames = compute_mfcc(self.audio_signal, self.sample_rate)
            
            self.log(f"✓ MFCC Extraction Complete!")
            self.log(f"  Frames: {len(self.mfcc_frames)}")
            self.log(f"  Features per frame: {len(self.mfcc_frames[0])}")
            
            self.visualize_mfcc()
            messagebox.showinfo("Success", "MFCC features extracted!")
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Error", str(e))
    
    def visualize_mfcc(self):
        for widget in self.mfcc_frame.winfo_children():
            widget.destroy()
        
        mfcc_T = [[self.mfcc_frames[t][d] for t in range(len(self.mfcc_frames))]
                  for d in range(len(self.mfcc_frames[0]))]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mfcc_T, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Feature Index')
        ax.set_title('MFCC Features')
        plt.colorbar(im, ax=ax)
        
        canvas = FigureCanvasTkAgg(fig, master=self.mfcc_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def acoustic_recognition(self):
        if self.mfcc_frames is None:
            messagebox.showwarning("Warning", "Extract MFCC features first!")
            return
        
        try:
            self.log("\n" + "="*60)
            self.log("ACOUSTIC PHONEME RECOGNITION")
            self.log("="*60)
            
            self.acoustic_sequence = [predict_phoneme(f) for f in self.mfcc_frames]
            
            self.log(f"✓ Acoustic Recognition Complete!")
            self.log(f"  Total frames: {len(self.acoustic_sequence)}")
            self.log(f"\nFirst 20 predictions:")
            self.log(f"  {self.acoustic_sequence[:20]}")
            
            messagebox.showinfo("Success", "Acoustic recognition complete!")
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Error", str(e))
    
    def process_all(self):
        if self.audio_signal is None:
            messagebox.showwarning("Warning", "Load audio file first!")
            return
        
        self.extract_mfcc()
        self.acoustic_recognition()

if __name__ == "__main__":
    root = tk.Tk()
    app = AcousticProcessingGUI(root)
    root.mainloop()
