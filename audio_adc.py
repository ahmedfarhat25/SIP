"""
Audio ADC (Analog to Digital Conversion) Script
Demonstrates sampling and quantization of audio signals.

Dependencies: numpy, scipy, soundfile, matplotlib
Install: pip install numpy scipy soundfile matplotlib
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioADC:
    """
    Audio Analog-to-Digital Conversion simulator.
    Demonstrates sampling and quantization effects on audio signals.
    """
    
    def __init__(self, audio_path: Optional[str] = None, 
                 signal_type: str = 'sine',
                 frequency: float = 440.0,
                 duration: float = 2.0,
                 original_sample_rate: int = 44100):
        """
        Initialize the ADC simulator.
        
        Args:
            audio_path: Path to audio file (if None, generates test signal)
            signal_type: Type of test signal ('sine', 'square', 'triangle', 'sawtooth')
            frequency: Frequency of test signal in Hz
            duration: Duration of test signal in seconds
            original_sample_rate: Sample rate for generated signals
        """
        self.original_sample_rate = original_sample_rate
        
        if audio_path:
            # Load audio file
            self.audio_path = Path(audio_path)
            self.original_audio, self.original_sample_rate = sf.read(str(self.audio_path))
            
            # Convert stereo to mono if needed
            if len(self.original_audio.shape) > 1:
                self.original_audio = np.mean(self.original_audio, axis=1)
            
            self.duration = len(self.original_audio) / self.original_sample_rate
            
            print(f"✓ Loaded audio: {self.audio_path.name}")
            print(f"  Sample rate: {self.original_sample_rate} Hz")
            print(f"  Duration: {self.duration:.2f} seconds")
        else:
            # Generate test signal
            self.duration = duration
            self.frequency = frequency
            self.signal_type = signal_type
            self.original_audio = self._generate_signal()
            
            print(f"✓ Generated {signal_type} wave")
            print(f"  Frequency: {frequency} Hz")
            print(f"  Duration: {duration} seconds")
            print(f"  Sample rate: {original_sample_rate} Hz")
        
        self.sampled_audio = None
        self.quantized_audio = None
    
    def _generate_signal(self) -> np.ndarray:
        """
        Generate test signal.
        
        Returns:
            Generated audio signal
        """
        t = np.linspace(0, self.duration, int(self.original_sample_rate * self.duration))
        omega = 2 * np.pi * self.frequency
        
        if self.signal_type == 'sine':
            signal = np.sin(omega * t)
        elif self.signal_type == 'square':
            signal = np.sign(np.sin(omega * t))
        elif self.signal_type == 'triangle':
            signal = 2 * np.arcsin(np.sin(omega * t)) / np.pi
        elif self.signal_type == 'sawtooth':
            signal = 2 * (t * self.frequency - np.floor(t * self.frequency + 0.5))
        else:
            raise ValueError(f"Unknown signal type: {self.signal_type}")
        
        # Normalize to [-1, 1]
        signal = signal / np.max(np.abs(signal))
        
        return signal
    
    def apply_sampling(self, target_sample_rate: int) -> np.ndarray:
        """
        Apply sampling (downsampling) to the audio.
        
        Args:
            target_sample_rate: Target sampling rate in Hz
            
        Returns:
            Sampled audio data
        """
        print(f"\n📊 Applying sampling...")
        print(f"  Original rate: {self.original_sample_rate} Hz")
        print(f"  Target rate: {target_sample_rate} Hz")
        
        # Calculate Nyquist frequency
        nyquist_freq = target_sample_rate / 2
        print(f"  Nyquist frequency: {nyquist_freq} Hz")
        
        # Downsample
        downsample_factor = self.original_sample_rate / target_sample_rate
        
        if downsample_factor < 1:
            print("  ⚠️ Warning: Target rate is higher than original rate")
            self.sampled_audio = self.original_audio.copy()
        else:
            # Simple decimation (take every nth sample)
            indices = np.arange(0, len(self.original_audio), downsample_factor).astype(int)
            self.sampled_audio = self.original_audio[indices]
            
            print(f"  ✓ Downsampled by factor of {downsample_factor:.2f}")
            print(f"  Original samples: {len(self.original_audio):,}")
            print(f"  Sampled samples: {len(self.sampled_audio):,}")
        
        return self.sampled_audio
    
    def apply_quantization(self, bit_depth: int, audio_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply quantization to the audio.
        
        Args:
            bit_depth: Number of bits for quantization (1-16)
            audio_data: Audio data to quantize (if None, uses sampled_audio or original_audio)
            
        Returns:
            Quantized audio data
        """
        if audio_data is None:
            audio_data = self.sampled_audio if self.sampled_audio is not None else self.original_audio
        
        print(f"\n🔢 Applying quantization...")
        print(f"  Bit depth: {bit_depth} bits")
        
        # Calculate number of quantization levels
        num_levels = 2 ** bit_depth
        print(f"  Quantization levels: {num_levels}")
        
        # Normalize to [-1, 1] if needed
        audio_normalized = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data
        
        # Quantize
        step_size = 2.0 / num_levels
        quantized = np.round(audio_normalized / step_size) * step_size
        
        # Clip to valid range
        quantized = np.clip(quantized, -1.0, 1.0)
        
        self.quantized_audio = quantized
        
        # Calculate quantization error
        quantization_error = np.mean(np.abs(audio_normalized - quantized))
        snr = self._calculate_snr(audio_normalized, quantized)
        
        print(f"  ✓ Quantization complete")
        print(f"  Step size: {step_size:.6f}")
        print(f"  Mean quantization error: {quantization_error:.6f}")
        print(f"  SNR: {snr:.2f} dB")
        
        return self.quantized_audio
    
    def _calculate_snr(self, original: np.ndarray, quantized: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio.
        
        Args:
            original: Original signal
            quantized: Quantized signal
            
        Returns:
            SNR in dB
        """
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - quantized) ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')
        
        return snr
    
    def reconstruct(self, target_sample_rate: int) -> np.ndarray:
        """
        Reconstruct signal from sampled/quantized data using interpolation.
        
        Args:
            target_sample_rate: Target sample rate for reconstruction
            
        Returns:
            Reconstructed audio signal
        """
        if self.quantized_audio is None:
            print("⚠️ Warning: No quantized audio available. Apply quantization first.")
            return None
        
        print(f"\n🔄 Reconstructing signal...")
        
        # Create time arrays
        original_time = np.linspace(0, self.duration, len(self.quantized_audio))
        reconstructed_time = np.linspace(0, self.duration, int(target_sample_rate * self.duration))
        
        # Linear interpolation
        reconstructed = np.interp(reconstructed_time, original_time, self.quantized_audio)
        
        print(f"  ✓ Reconstructed to {target_sample_rate} Hz")
        print(f"  Samples: {len(reconstructed):,}")
        
        return reconstructed
    
    def visualize(self, save_path: Optional[str] = None, show_plot: bool = True):
        """
        Visualize the ADC process.
        
        Args:
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
        """
        print(f"\n📈 Creating visualization...")
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        fig.suptitle('Audio ADC Process: Sampling & Quantization', fontsize=16, fontweight='bold')
        
        # Determine time window to display (first 0.1 seconds or less)
        display_duration = min(0.1, self.duration)
        display_samples_original = int(display_duration * self.original_sample_rate)
        
        time_original = np.linspace(0, display_duration, display_samples_original)
        
        # Plot 1: Original Signal
        axes[0].plot(time_original, self.original_audio[:display_samples_original], 
                    'b-', linewidth=1.5, label='Original Signal')
        axes[0].set_title('1. Original Analog Signal', fontweight='bold')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_xlim(0, display_duration)
        
        # Plot 2: Sampled Signal
        if self.sampled_audio is not None:
            # Calculate corresponding sample rate
            sample_rate_ratio = len(self.sampled_audio) / len(self.original_audio)
            display_samples_sampled = int(display_samples_original * sample_rate_ratio)
            time_sampled = np.linspace(0, display_duration, display_samples_sampled)
            
            axes[1].plot(time_original, self.original_audio[:display_samples_original], 
                        'b-', linewidth=0.5, alpha=0.3, label='Original')
            axes[1].stem(time_sampled, self.sampled_audio[:display_samples_sampled], 
                        linefmt='purple', markerfmt='o', basefmt=' ', label='Samples')
            axes[1].set_title('2. Sampled Signal (Discrete Time)', fontweight='bold')
            axes[1].set_ylabel('Amplitude')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            axes[1].set_xlim(0, display_duration)
        
        # Plot 3: Quantized Signal
        if self.quantized_audio is not None:
            sample_rate_ratio = len(self.quantized_audio) / len(self.original_audio)
            display_samples_quantized = int(display_samples_original * sample_rate_ratio)
            time_quantized = np.linspace(0, display_duration, display_samples_quantized)
            
            axes[2].plot(time_original, self.original_audio[:display_samples_original], 
                        'b-', linewidth=0.5, alpha=0.3, label='Original')
            axes[2].plot(time_quantized, self.quantized_audio[:display_samples_quantized], 
                        'r-', linewidth=1.5, drawstyle='steps-post', label='Quantized')
            axes[2].set_title('3. Quantized Signal (Discrete Amplitude)', fontweight='bold')
            axes[2].set_ylabel('Amplitude')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()
            axes[2].set_xlim(0, display_duration)
        
        # Plot 4: Comparison
        if self.quantized_audio is not None:
            sample_rate_ratio = len(self.quantized_audio) / len(self.original_audio)
            display_samples_quantized = int(display_samples_original * sample_rate_ratio)
            time_quantized = np.linspace(0, display_duration, display_samples_quantized)
            
            axes[3].plot(time_original, self.original_audio[:display_samples_original], 
                        'b-', linewidth=1, alpha=0.5, label='Original')
            axes[3].plot(time_quantized, self.quantized_audio[:display_samples_quantized], 
                        'g-', linewidth=1.5, label='Digitized (Sampled + Quantized)')
            axes[3].set_title('4. Original vs Digitized Signal', fontweight='bold')
            axes[3].set_xlabel('Time (seconds)')
            axes[3].set_ylabel('Amplitude')
            axes[3].grid(True, alpha=0.3)
            axes[3].legend()
            axes[3].set_xlim(0, display_duration)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved plot: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def save_audio(self, output_path: str, sample_rate: Optional[int] = None):
        """
        Save processed audio to file.
        
        Args:
            output_path: Output file path
            sample_rate: Sample rate for output (default: uses original)
        """
        if self.quantized_audio is None:
            print("⚠️ Warning: No processed audio to save")
            return
        
        if sample_rate is None:
            sample_rate = self.original_sample_rate
        
        sf.write(output_path, self.quantized_audio, sample_rate)
        print(f"\n💾 Saved audio: {output_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Audio ADC: Sampling and Quantization Demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process audio file with custom sampling and quantization
  python audio_adc.py input.wav --sample-rate 8000 --bit-depth 8
  
  # Generate and process sine wave
  python audio_adc.py --generate sine --frequency 440 --sample-rate 8000 --bit-depth 4
  
  # Generate square wave and visualize
  python audio_adc.py --generate square --frequency 100 --visualize
        """
    )
    
    parser.add_argument('input', type=str, nargs='?', help='Input audio file path')
    parser.add_argument('--generate', type=str, choices=['sine', 'square', 'triangle', 'sawtooth'],
                       help='Generate test signal instead of loading file')
    parser.add_argument('--frequency', type=float, default=440,
                       help='Frequency for generated signal (default: 440 Hz)')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='Duration for generated signal (default: 2.0 seconds)')
    parser.add_argument('--sample-rate', type=int, default=8000,
                       help='Target sampling rate in Hz (default: 8000)')
    parser.add_argument('--bit-depth', type=int, default=8,
                       help='Bit depth for quantization (default: 8)')
    parser.add_argument('-o', '--output', type=str,
                       help='Output audio file path')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plot')
    parser.add_argument('--save-plot', type=str,
                       help='Save plot to file')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 60)
    print("🎵 AUDIO ADC: SAMPLING & QUANTIZATION")
    print("=" * 60)
    
    # Initialize ADC
    if args.generate:
        adc = AudioADC(signal_type=args.generate, frequency=args.frequency, duration=args.duration)
    elif args.input:
        adc = AudioADC(audio_path=args.input)
    else:
        parser.error("Either provide input file or use --generate")
    
    # Apply sampling
    adc.apply_sampling(args.sample_rate)
    
    # Apply quantization
    adc.apply_quantization(args.bit_depth)
    
    # Save output
    if args.output:
        adc.save_audio(args.output, args.sample_rate)
    
    # Visualize
    if args.visualize or args.save_plot:
        adc.visualize(save_path=args.save_plot, show_plot=args.visualize)
    
    print("\n" + "=" * 60)
    print("✅ ADC processing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
