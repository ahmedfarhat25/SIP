"""
Audio Preprocessing Script
Implements noise reduction and silence removal for audio files.

Dependencies: numpy, scipy, soundfile
Install: pip install numpy scipy soundfile matplotlib
"""

import numpy as np
import soundfile as sf
import argparse
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """
    Audio preprocessing class for noise reduction and silence removal.
    """
    
    def __init__(self, audio_path: str):
        """
        Initialize the audio preprocessor.
        
        Args:
            audio_path: Path to the input audio file
        """
        self.audio_path = Path(audio_path)
        self.audio_data, self.sample_rate = sf.read(str(self.audio_path))
        
        # Convert stereo to mono if needed
        if len(self.audio_data.shape) > 1:
            self.audio_data = np.mean(self.audio_data, axis=1)
        
        self.processed_audio = self.audio_data.copy()
        
        print(f"✓ Loaded audio: {self.audio_path.name}")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Duration: {len(self.audio_data) / self.sample_rate:.2f} seconds")
        print(f"  Samples: {len(self.audio_data)}")
    
    def apply_noise_reduction(self, noise_threshold_db: float = -40) -> np.ndarray:
        """
        Apply noise reduction using spectral gating.
        
        Args:
            noise_threshold_db: Threshold in dB below which signal is considered noise
            
        Returns:
            Processed audio data
        """
        print(f"\n🔇 Applying noise reduction (threshold: {noise_threshold_db} dB)...")
        
        # Convert dB to linear amplitude
        threshold_linear = 10 ** (noise_threshold_db / 20)
        
        # Apply noise gate
        audio_abs = np.abs(self.processed_audio)
        noise_mask = audio_abs < threshold_linear
        
        # Reduce noise by 90% instead of complete removal for smoother result
        self.processed_audio[noise_mask] *= 0.1
        
        noise_samples = np.sum(noise_mask)
        noise_percentage = (noise_samples / len(self.processed_audio)) * 100
        
        print(f"  ✓ Reduced {noise_samples} samples ({noise_percentage:.1f}%)")
        
        return self.processed_audio
    
    def remove_silence(self, 
                      silence_threshold_db: float = -50,
                      min_silence_duration: float = 0.5) -> np.ndarray:
        """
        Remove silent portions from audio.
        
        Args:
            silence_threshold_db: Threshold in dB below which audio is considered silent
            min_silence_duration: Minimum duration of silence to remove (seconds)
            
        Returns:
            Processed audio data with silence removed
        """
        print(f"\n✂️ Removing silence (threshold: {silence_threshold_db} dB, "
              f"min duration: {min_silence_duration}s)...")
        
        # Convert dB to linear amplitude
        threshold_linear = 10 ** (silence_threshold_db / 20)
        min_silence_samples = int(min_silence_duration * self.sample_rate)
        
        # Detect silence
        audio_abs = np.abs(self.processed_audio)
        is_silent = audio_abs < threshold_linear
        
        # Find continuous segments
        segments = []
        in_silence = is_silent[0]
        segment_start = 0
        
        for i in range(1, len(is_silent)):
            if is_silent[i] != in_silence:
                # Segment boundary
                if not in_silence:
                    # End of non-silent segment
                    segments.append((segment_start, i))
                
                segment_start = i
                in_silence = is_silent[i]
        
        # Add final segment if non-silent
        if not in_silence:
            segments.append((segment_start, len(is_silent)))
        
        # Filter out very short non-silent segments between long silences
        filtered_segments = []
        for start, end in segments:
            duration_samples = end - start
            if duration_samples > min_silence_samples // 10:  # Keep segments > 10% of min silence
                filtered_segments.append((start, end))
        
        # Concatenate non-silent segments
        if filtered_segments:
            audio_segments = [self.processed_audio[start:end] for start, end in filtered_segments]
            self.processed_audio = np.concatenate(audio_segments)
        else:
            print("  ⚠️ Warning: No non-silent segments found!")
            self.processed_audio = self.processed_audio[:1000]  # Keep minimal audio
        
        original_duration = len(self.audio_data) / self.sample_rate
        new_duration = len(self.processed_audio) / self.sample_rate
        removed_duration = original_duration - new_duration
        reduction_percent = (removed_duration / original_duration) * 100
        
        print(f"  ✓ Removed {removed_duration:.2f}s of silence ({reduction_percent:.1f}%)")
        print(f"  Original: {original_duration:.2f}s → Processed: {new_duration:.2f}s")
        
        return self.processed_audio
    
    def normalize(self, target_db: float = -3.0) -> np.ndarray:
        """
        Normalize audio to target dB level.
        
        Args:
            target_db: Target peak level in dB
            
        Returns:
            Normalized audio data
        """
        print(f"\n🔊 Normalizing audio to {target_db} dB...")
        
        # Find current peak
        current_peak = np.max(np.abs(self.processed_audio))
        
        if current_peak > 0:
            # Calculate target amplitude
            target_amplitude = 10 ** (target_db / 20)
            
            # Apply normalization
            self.processed_audio = self.processed_audio * (target_amplitude / current_peak)
            
            print(f"  ✓ Normalized (peak: {current_peak:.4f} → {target_amplitude:.4f})")
        else:
            print("  ⚠️ Warning: Audio is silent, skipping normalization")
        
        return self.processed_audio
    
    def save(self, output_path: Optional[str] = None) -> str:
        """
        Save processed audio to file.
        
        Args:
            output_path: Output file path (default: adds '_processed' to input filename)
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = self.audio_path.parent / f"{self.audio_path.stem}_processed.wav"
        else:
            output_path = Path(output_path)
        
        sf.write(str(output_path), self.processed_audio, self.sample_rate)
        
        print(f"\n💾 Saved processed audio: {output_path.name}")
        
        return str(output_path)
    
    def get_statistics(self) -> dict:
        """
        Get statistics about original and processed audio.
        
        Returns:
            Dictionary with audio statistics
        """
        original_duration = len(self.audio_data) / self.sample_rate
        processed_duration = len(self.processed_audio) / self.sample_rate
        
        return {
            'original_duration': original_duration,
            'processed_duration': processed_duration,
            'reduction_percent': ((original_duration - processed_duration) / original_duration) * 100,
            'original_samples': len(self.audio_data),
            'processed_samples': len(self.processed_audio),
            'sample_rate': self.sample_rate,
            'original_peak': np.max(np.abs(self.audio_data)),
            'processed_peak': np.max(np.abs(self.processed_audio))
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Audio Preprocessing: Noise Reduction and Silence Removal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python audio_preprocessing.py input.wav
  
  # Custom noise and silence thresholds
  python audio_preprocessing.py input.wav --noise-threshold -35 --silence-threshold -45
  
  # Specify output file
  python audio_preprocessing.py input.wav -o output.wav
  
  # Apply only noise reduction
  python audio_preprocessing.py input.wav --no-silence-removal
        """
    )
    
    parser.add_argument('input', type=str, help='Input audio file path')
    parser.add_argument('-o', '--output', type=str, help='Output audio file path')
    parser.add_argument('--noise-threshold', type=float, default=-40,
                       help='Noise threshold in dB (default: -40)')
    parser.add_argument('--silence-threshold', type=float, default=-50,
                       help='Silence threshold in dB (default: -50)')
    parser.add_argument('--min-silence-duration', type=float, default=0.5,
                       help='Minimum silence duration in seconds (default: 0.5)')
    parser.add_argument('--no-noise-reduction', action='store_true',
                       help='Skip noise reduction')
    parser.add_argument('--no-silence-removal', action='store_true',
                       help='Skip silence removal')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize audio after processing')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 60)
    print("🎵 AUDIO PREPROCESSING")
    print("=" * 60)
    
    # Load audio
    preprocessor = AudioPreprocessor(args.input)
    
    # Apply processing
    if not args.no_noise_reduction:
        preprocessor.apply_noise_reduction(args.noise_threshold)
    
    if not args.no_silence_removal:
        preprocessor.remove_silence(args.silence_threshold, args.min_silence_duration)
    
    if args.normalize:
        preprocessor.normalize()
    
    # Save output
    output_path = preprocessor.save(args.output)
    
    # Print statistics
    stats = preprocessor.get_statistics()
    print("\n" + "=" * 60)
    print("📊 STATISTICS")
    print("=" * 60)
    print(f"Original Duration:   {stats['original_duration']:.2f}s")
    print(f"Processed Duration:  {stats['processed_duration']:.2f}s")
    print(f"Reduction:           {stats['reduction_percent']:.1f}%")
    print(f"Original Samples:    {stats['original_samples']:,}")
    print(f"Processed Samples:   {stats['processed_samples']:,}")
    print(f"Sample Rate:         {stats['sample_rate']:,} Hz")
    print(f"Original Peak:       {stats['original_peak']:.4f}")
    print(f"Processed Peak:      {stats['processed_peak']:.4f}")
    print("=" * 60)
    print(f"✅ Processing complete! Output: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
