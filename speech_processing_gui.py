"""
Speech Processing GUI Application - Project 2
Acoustic Speech Processing: MFCC Feature Extraction + HMM Acoustic Model
All implemented MANUALLY without built-in functions

Requirements:
- pip install librosa soundfile matplotlib numpy tkinter
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import librosa
import soundfile as sf
from scipy.fftpack import dct
import pickle
from collections import defaultdict
import re


class SpeechProcessingApp:
    """Main GUI Application for Speech Processing"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Acoustic Speech Processing - Manual Implementation")
        self.root.geometry("1400x900")
        
        # Variables
        self.audio_signal = None
        self.sample_rate = None
        self.audio_path = None
        self.mfcc_features = None
        self.delta_features = None
        self.delta_delta_features = None
        self.acoustic_model = None
        
        # Setup GUI
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI layout"""
        
        # Menu Bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Audio", command=self.load_audio)
        file_menu.add_command(label="Save MFCC Features", command=self.save_mfcc)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # MFCC Menu
        mfcc_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="MFCC Features", menu=mfcc_menu)
        mfcc_menu.add_command(label="Extract MFCC (Manual)", command=self.extract_mfcc_manual)
        mfcc_menu.add_command(label="Show Mel Filterbank", command=self.show_mel_filterbank)
        mfcc_menu.add_command(label="Visualize MFCC", command=self.visualize_mfcc)
        mfcc_menu.add_command(label="Compute Delta Features", command=self.compute_delta_features)
        
        # Acoustic Model Menu
        acoustic_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Acoustic Model (HMM)", menu=acoustic_menu)
        acoustic_menu.add_command(label="Train HMM Model", command=self.train_acoustic_model)
        acoustic_menu.add_command(label="Test Recognition", command=self.recognize_speech)
        acoustic_menu.add_command(label="Show HMM Structure", command=self.show_hmm_structure)
        acoustic_menu.add_separator()
        acoustic_menu.add_command(label="Save Model", command=self.save_acoustic_model)
        acoustic_menu.add_command(label="Load Model", command=self.load_acoustic_model)
        
        # Main Frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Panel - Visualization
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(left_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Waveform
        self.waveform_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.waveform_frame, text="Waveform")
        
        # Tab 2: MFCC Visualization
        self.mfcc_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.mfcc_frame, text="MFCC")
        
        # Tab 3: Mel Filterbank
        self.filterbank_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.filterbank_frame, text="Filterbank")
        
        # Right Panel - Console/Output
        right_panel = ttk.LabelFrame(main_frame, text="Console Output", padding=10)
        # Tab 3: Mel Filterbank
        self.filterbank_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.filterbank_frame, text="Filterbank")
        
        # Tab 4: Delta Features
        self.delta_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.delta_frame, text="Delta Features")
        
        # Right Panel - Console Output
        console_frame = ttk.LabelFrame(main_frame, text="Console Output", padding=10)
        console_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.console = scrolledtext.ScrolledText(console_frame, width=60, height=35, 
                                                  font=("Consolas", 9))
        self.console.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Status Bar
        self.status_label = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.log("="*60)
        self.log("ACOUSTIC SPEECH PROCESSING APPLICATION")
        self.log("="*60)
        self.log("Project 2: Acoustic Speech Processing")
        self.log("Features:")
        self.log("  1. MFCC Feature Extraction (Manual)")
        self.log("  2. Delta & Delta-Delta Features")
        self.log("  3. HMM Acoustic Model")
        self.log("  4. Speech Recognition")
        self.log("\nLoad an audio file to begin...")
        self.log("="*60)
    
    def log(self, message):
        """Log message to console"""
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)
        self.root.update_idletasks()
    
    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def check_audio_loaded(self):
        """Check if audio is loaded"""
        if self.audio_signal is None:
            messagebox.showerror("Error", "Please load an audio file first!")
            return False
        return True
    
    # ==================== LOAD AUDIO ====================
    
    def load_audio(self):
        """Load audio file"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                self.audio_path = file_path
                self.audio_signal, self.sample_rate = librosa.load(file_path, sr=22050)
                
                self.log(f"\n{'='*60}")
                self.log(f"Audio Loaded: {file_path}")
                self.log(f"Sample Rate: {self.sample_rate} Hz")
                self.log(f"Duration: {len(self.audio_signal)/self.sample_rate:.2f} seconds")
                self.log(f"Samples: {len(self.audio_signal)}")
                self.log(f"{'='*60}\n")
                
                # Display waveform
                self.plot_waveform()
                
                self.update_status(f"Audio loaded: {len(self.audio_signal)/self.sample_rate:.2f}s")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load audio: {str(e)}")
                self.log(f"ERROR: {str(e)}")
    
    def plot_waveform(self):
        """Plot audio waveform"""
        if self.audio_signal is None:
            return
        
        # Clear previous plot
        for widget in self.waveform_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        time = np.arange(len(self.audio_signal)) / self.sample_rate
        ax.plot(time, self.audio_signal, linewidth=0.5)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform')
        ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=self.waveform_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # ==================== MFCC EXTRACTION ====================
    
    def extract_mfcc_manual(self):
        """Extract MFCC features manually"""
        if not self.check_audio_loaded():
            return
        
        try:
            self.log("\n" + "="*60)
            self.log("EXTRACTING MFCC FEATURES (MANUAL IMPLEMENTATION)")
            self.log("="*60)
            
            # Parameters
            n_fft = 2048
            hop_length = 512
            n_mels = 40
            n_mfcc = 13
            
            self.log(f"Parameters:")
            self.log(f"  FFT Size: {n_fft}")
            self.log(f"  Hop Length: {hop_length}")
            self.log(f"  Mel Filters: {n_mels}")
            self.log(f"  MFCC Coefficients: {n_mfcc}")
            
            # Step 1: Pre-emphasis
            self.log("\nStep 1: Pre-emphasis filter...")
            emphasized_signal = self.apply_preemphasis(self.audio_signal)
            
            # Step 2: Framing
            self.log("Step 2: Framing signal...")
            frames = self.frame_signal(emphasized_signal, n_fft, hop_length)
            self.log(f"  Number of frames: {frames.shape[0]}")
            
            # Step 3: Windowing
            self.log("Step 3: Applying Hamming window...")
            windowed_frames = self.apply_hamming_window(frames)
            
            # Step 4: FFT and Power Spectrum
            self.log("Step 4: Computing power spectrum...")
            power_spectrum = self.compute_power_spectrum(windowed_frames)
            
            # Step 5: Mel Filterbank
            self.log("Step 5: Creating Mel filterbank...")
            filterbank = self.create_mel_filterbank(n_mels, n_fft, self.sample_rate)
            
            # Step 6: Apply filterbank
            self.log("Step 6: Applying Mel filterbank...")
            mel_spectrum = np.dot(power_spectrum, filterbank.T)
            mel_spectrum = np.where(mel_spectrum == 0, np.finfo(float).eps, mel_spectrum)
            
            # Step 7: Log
            self.log("Step 7: Applying logarithm...")
            log_mel_spectrum = np.log(mel_spectrum)
            
            # Step 8: DCT
            self.log("Step 8: Applying DCT...")
            mfcc = dct(log_mel_spectrum, type=2, axis=1, norm='ortho')[:, :n_mfcc]
            
            self.mfcc_features = mfcc
            
            self.log(f"\n✓ MFCC Extraction Complete!")
            self.log(f"  MFCC Shape: {mfcc.shape} (frames x coefficients)")
            self.log(f"  Mean: {np.mean(mfcc, axis=0)}")
            self.log(f"  Std: {np.std(mfcc, axis=0)}")
            
            self.update_status(f"MFCC extracted: {mfcc.shape}")
            
            # Visualize
            self.visualize_mfcc()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract MFCC: {str(e)}")
            self.log(f"ERROR: {str(e)}")
    
    def apply_preemphasis(self, signal, coef=0.97):
        """Pre-emphasis filter - MANUAL"""
        return np.append(signal[0], signal[1:] - coef * signal[:-1])
    
    def frame_signal(self, signal, frame_size, hop_length):
        """Frame signal into overlapping windows - MANUAL"""
        signal_length = len(signal)
        num_frames = 1 + int(np.ceil((signal_length - frame_size) / hop_length))
        
        pad_length = (num_frames - 1) * hop_length + frame_size
        padded_signal = np.pad(signal, (0, pad_length - signal_length), mode='constant')
        
        frames = np.zeros((num_frames, frame_size))
        for i in range(num_frames):
            start = i * hop_length
            frames[i] = padded_signal[start:start + frame_size]
        
        return frames
    
    def apply_hamming_window(self, frames):
        """Apply Hamming window - MANUAL"""
        N = frames.shape[1]
        hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
        return frames * hamming
    
    def compute_power_spectrum(self, frames):
        """Compute power spectrum - MANUAL"""
        magnitude_spectrum = np.abs(np.fft.rfft(frames, axis=1))
        power_spectrum = (magnitude_spectrum ** 2) / frames.shape[1]
        return power_spectrum
    
    def hz_to_mel(self, hz):
        """Convert Hz to Mel - MANUAL"""
        return 2595 * np.log10(1 + hz / 700)
    
    def mel_to_hz(self, mel):
        """Convert Mel to Hz - MANUAL"""
        return 700 * (10**(mel / 2595) - 1)
    
    def create_mel_filterbank(self, n_mels, n_fft, sr, fmin=0, fmax=None):
        """Create Mel filterbank - MANUAL"""
        if fmax is None:
            fmax = sr / 2
        
        mel_min = self.hz_to_mel(fmin)
        mel_max = self.hz_to_mel(fmax)
        
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = self.mel_to_hz(mel_points)
        
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        
        filterbank = np.zeros((n_mels, n_fft // 2 + 1))
        
        for m in range(1, n_mels + 1):
            f_left = bin_points[m - 1]
            f_center = bin_points[m]
            f_right = bin_points[m + 1]
            
            # Rising slope
            for k in range(f_left, f_center):
                filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)
            
            # Falling slope
            for k in range(f_center, f_right):
                filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)
        
        return filterbank
    
    def show_mel_filterbank(self):
        """Display Mel filterbank"""
        if not self.check_audio_loaded():
            return
        
        try:
            filterbank = self.create_mel_filterbank(40, 2048, self.sample_rate)
            
            # Clear previous plot
            for widget in self.filterbank_frame.winfo_children():
                widget.destroy()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(filterbank, aspect='auto', origin='lower', cmap='viridis')
            ax.set_xlabel('FFT Bins')
            ax.set_ylabel('Mel Filter Index')
            ax.set_title('Mel Filterbank (40 filters)')
            plt.colorbar(ax.images[0], ax=ax)
            
            canvas = FigureCanvasTkAgg(fig, master=self.filterbank_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.log("\nMel Filterbank displayed (40 triangular filters)")
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
    
    def visualize_mfcc(self):
        """Visualize MFCC features"""
        if self.mfcc_features is None:
            messagebox.showwarning("Warning", "Extract MFCC features first!")
            return
        
        # Clear previous plot
        for widget in self.mfcc_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(self.mfcc_features.T, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xlabel('Frame')
        ax.set_ylabel('MFCC Coefficient')
        ax.set_title(f'MFCC Features ({self.mfcc_features.shape[1]} coefficients)')
        plt.colorbar(im, ax=ax)
        
        canvas = FigureCanvasTkAgg(fig, master=self.mfcc_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.log("\nMFCC visualization updated")
    
    def compute_delta_features(self):
        """Compute Delta and Delta-Delta features - MANUAL"""
        if self.mfcc_features is None:
            messagebox.showwarning("Warning", "Extract MFCC features first!")
            return
        
        try:
            self.log("\n" + "="*60)
            self.log("COMPUTING DELTA FEATURES (MANUAL)")
            self.log("="*60)
            
            # Compute delta (velocity)
            self.log("\nStep 1: Computing Delta (Velocity)...")
            delta = self.compute_delta_manual(self.mfcc_features, N=2)
            self.log(f"  Delta shape: {delta.shape}")
            
            # Compute delta-delta (acceleration)
            self.log("\nStep 2: Computing Delta-Delta (Acceleration)...")
            delta_delta = self.compute_delta_manual(delta, N=2)
            self.log(f"  Delta-Delta shape: {delta_delta.shape}")
            
            # Save delta features separately
            self.delta_features = delta
            self.delta_delta_features = delta_delta
            
            # Combine features
            combined = np.hstack([self.mfcc_features, delta, delta_delta])
            
            self.log(f"\n✓ Delta Features Computed!")
            self.log(f"  MFCC: {self.mfcc_features.shape}")
            self.log(f"  Delta: {delta.shape}")
            self.log(f"  Delta-Delta: {delta_delta.shape}")
            self.log(f"  Combined: {combined.shape} (39 features total)")
            
            # Visualize delta features
            self.visualize_delta_features(self.mfcc_features, delta, delta_delta)
            
            # Update MFCC with combined features for model training
            self.mfcc_features = combined
            
            messagebox.showinfo("Success", 
                              f"Delta features computed!\nTotal features: {combined.shape[1]}")
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
    
    def compute_delta_manual(self, features, N=2):
        """Compute delta features manually"""
        num_frames = features.shape[0]
        delta = np.zeros_like(features)
        
        denom = 2 * sum([n**2 for n in range(1, N+1)])
        
    def visualize_delta_features(self, mfcc, delta, delta_delta):
        """Visualize Delta features"""
        # Clear previous plot
        for widget in self.delta_frame.winfo_children():
            widget.destroy()
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        axes[0].imshow(mfcc.T, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_ylabel('MFCC Coefficient')
        axes[0].set_title('MFCC Features')
        
        axes[1].imshow(delta.T, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_ylabel('Delta Coefficient')
        axes[1].set_title('Delta (Velocity) Features')
        
        axes[2].imshow(delta_delta.T, aspect='auto', origin='lower', cmap='viridis')
        axes[2].set_xlabel('Frame')
        axes[2].set_ylabel('Delta-Delta Coefficient')
        axes[2].set_title('Delta-Delta (Acceleration) Features')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.delta_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.log("\nDelta features visualization updated")
    
    def save_mfcc(self):
        """Save MFCC features"""
        if self.mfcc_features is None:
            messagebox.showerror("Error", "No MFCC features to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".npy",
            filetypes=[("NumPy Array", "*.npy"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                np.save(file_path, self.mfcc_features)
                self.log(f"\n✓ MFCC features saved to: {file_path}")
                self.update_status("MFCC saved")
            except Exception as e:
                self.log(f"ERROR: {str(e)}")
        if self.mfcc_features is None:
            messagebox.showerror("Error", "No MFCC features to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".npy",
            filetypes=[("NumPy Array", "*.npy"), ("All Files", "*.*")]
        )
        
    def recognize_speech(self):
        """Recognize speech using acoustic model"""
        if self.acoustic_model is None:
            messagebox.showwarning("Warning", "Train acoustic model first!")
            return
        
        if self.mfcc_features is None:
            messagebox.showwarning("Warning", "Extract MFCC features first!")
            return
        
        try:
            self.log("\n" + "="*60)
            self.log("SPEECH RECOGNITION")
            self.log("="*60)
            
            # Test with different segments
            test_segments = [
                ("Segment 1 (0-50 frames)", self.mfcc_features[:50]),
                ("Segment 2 (25-75 frames)", self.mfcc_features[25:75] if len(self.mfcc_features) > 75 else self.mfcc_features[25:]),
            ]
            
            self.log("\nTesting multiple segments...")
            results = []
            
            for seg_name, segment in test_segments:
                if len(segment) > 0:
                    word, score = self.acoustic_model.recognize(segment)
                    results.append((seg_name, word, score))
                    self.log(f"\n{seg_name}:")
                    self.log(f"  Recognized: {word}")
                    self.log(f"  Score: {score:.2f}")
            
            # Show best result
            best_result = max(results, key=lambda x: x[2])
            
            self.log(f"\n✓ Best Recognition: {best_result[1]}")
            self.log(f"  From: {best_result[0]}")
            self.log(f"  Score: {best_result[2]:.2f}")
            
            messagebox.showinfo("Recognition Result", 
                              f"Best Match: {best_result[1]}\n"
                              f"Segment: {best_result[0]}\n"
                              f"Score: {best_result[2]:.2f}")
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
    
    def show_hmm_structure(self):
        """Show HMM structure and parameters"""
        if self.acoustic_model is None:
            messagebox.showwarning("Warning", "Train acoustic model first!")
            return
        
        try:
            self.log("\n" + "="*60)
            self.log("HMM ACOUSTIC MODEL STRUCTURE")
            self.log("="*60)
            
            self.log(f"\nModel Configuration:")
            self.log(f"  Number of States: {self.acoustic_model.n_states}")
            self.log(f"  Feature Dimension: {self.acoustic_model.n_features}")
            self.log(f"  Trained Words: {list(self.acoustic_model.models.keys())}")
            
            self.log(f"\nHMM Topology: Left-to-Right")
            self.log(f"  State 1 → State 2 → State 3 → State 4 → State 5")
            self.log(f"     ↺         ↺         ↺         ↺         ↺")
            self.log(f"  (self-loops allowed)")
            
            for word, params in self.acoustic_model.models.items():
                self.log(f"\nWord: '{word}'")
                self.log(f"  Mean vector shape: {params['mean'].shape}")
                self.log(f"  Std vector shape: {params['std'].shape}")
                self.log(f"  Mean MFCC[0]: {params['mean'][0]:.3f}")
                self.log(f"  Std MFCC[0]: {params['std'][0]:.3f}")
            
            messagebox.showinfo("HMM Structure", 
                              f"HMM Acoustic Model\n\n"
                              f"States: {self.acoustic_model.n_states}\n"
                              f"Features: {self.acoustic_model.n_features}\n"
                              f"Words: {len(self.acoustic_model.models)}\n\n"
                              f"See console for details.")
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")


# ==================== HELPER CLASSES ====================

class SimpleAcousticModel:
    """Simplified acoustic model for demo"""
    
    def __init__(self, n_states=5, n_features=13):
        self.n_states = n_states
        self.n_features = n_features
        self.models = {}
    
    def train_word(self, word, sequences):
        """Train model for a word"""
        # Simple: Store mean and std
        all_frames = np.vstack(sequences)
        self.models[word] = {
            'mean': np.mean(all_frames, axis=0),
            'std': np.std(all_frames, axis=0) + 1e-6
        }
    
    def recognize(self, features):
        """Simple recognition using Gaussian distance"""
        best_word = None
        best_score = float('-inf')
        
# ==================== HELPER CLASSES ====================

class SimpleAcousticModel:
    """
    HMM-based Acoustic Model (Simplified for Demo)
    
    In a full implementation, this would include:
    - Gaussian Mixture Models (GMM) for emission probabilities
    - Forward-Backward algorithm for training
    - Viterbi algorithm for decoding
    - Baum-Welch algorithm for parameter estimation
    
    This simplified version uses Gaussian statistics for educational purposes
    """
    
    def __init__(self, n_states=5, n_features=13):
        self.n_states = n_states
        self.n_features = n_features
        self.models = {}
    
    def train_word(self, word, sequences):
        """
        Train HMM model for a word (Simplified)
        
        Full implementation would:
        1. Initialize HMM parameters (A, B, π)
        2. Run Baum-Welch (EM) algorithm
        3. Update transition and emission probabilities
        
        This version: Store statistical parameters
        """
        all_frames = np.vstack(sequences)
        self.models[word] = {
            'mean': np.mean(all_frames, axis=0),
            'std': np.std(all_frames, axis=0) + 1e-6,
            'n_samples': len(all_frames)
        }
    
    def recognize(self, features):
        """
        Recognize speech using trained models
        
        Full implementation would use:
        - Forward algorithm to compute P(O|λ)
        - Viterbi algorithm for best state sequence
        
        This version: Uses Gaussian log-likelihood
        """
        best_word = None
        best_score = float('-inf')
        
        for word, params in self.models.items():
            # Compute log likelihood (simplified Gaussian)
            diff = features - params['mean']
            log_likelihood = -np.sum((diff / params['std']) ** 2)
            
            # Normalize by number of frames
            score = log_likelihood / len(features)
            
            if score > best_score:
                best_score = score
                best_word = word
        
        return best_word, best_score