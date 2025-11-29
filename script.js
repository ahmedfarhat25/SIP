// ===========================
// ADC SIMULATION MODULE
// ===========================

class ADCSimulator {
  constructor() {
    this.analogCanvas = document.getElementById('analogCanvas');
    this.sampledCanvas = document.getElementById('sampledCanvas');
    this.quantizedCanvas = document.getElementById('quantizedCanvas');
    this.reconstructedCanvas = document.getElementById('reconstructedCanvas');
    
    this.signalType = 'sine';
    this.frequency = 5;
    this.sampleRate = 50;
    this.bitDepth = 8;
    
    this.setupEventListeners();
    this.animate();
  }
  
  setupEventListeners() {
    document.getElementById('signalType').addEventListener('change', (e) => {
      this.signalType = e.target.value;
      this.draw();
    });
    
    document.getElementById('frequency').addEventListener('input', (e) => {
      this.frequency = parseFloat(e.target.value);
      document.getElementById('freqValue').textContent = `${this.frequency} Hz`;
      this.updateStats();
      this.draw();
    });
    
    document.getElementById('sampleRate').addEventListener('input', (e) => {
      this.sampleRate = parseFloat(e.target.value);
      document.getElementById('sampleRateValue').textContent = `${this.sampleRate} Hz`;
      this.updateStats();
      this.draw();
    });
    
    document.getElementById('bitDepth').addEventListener('input', (e) => {
      this.bitDepth = parseInt(e.target.value);
      document.getElementById('bitDepthValue').textContent = `${this.bitDepth} bits`;
      this.updateStats();
      this.draw();
    });
  }
  
  updateStats() {
    const nyquist = this.frequency * 2;
    const quantLevels = Math.pow(2, this.bitDepth);
    const bitRate = this.sampleRate * this.bitDepth;
    
    document.getElementById('nyquistRate').textContent = `${nyquist} Hz`;
    document.getElementById('quantLevels').textContent = quantLevels;
    document.getElementById('bitRate').textContent = `${bitRate} bps`;
  }
  
  generateSignal(t) {
    const omega = 2 * Math.PI * this.frequency;
    
    switch(this.signalType) {
      case 'sine':
        return Math.sin(omega * t);
      case 'square':
        return Math.sign(Math.sin(omega * t));
      case 'triangle':
        return (2 / Math.PI) * Math.asin(Math.sin(omega * t));
      case 'sawtooth':
        return 2 * (t * this.frequency - Math.floor(t * this.frequency + 0.5));
      default:
        return Math.sin(omega * t);
    }
  }
  
  drawSignal(canvas, drawFunc) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = '#0a0e27';
    ctx.fillRect(0, 0, width, height);
    
    // Draw grid
    this.drawGrid(ctx, width, height);
    
    // Draw signal
    drawFunc(ctx, width, height);
  }
  
  drawGrid(ctx, width, height) {
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.1)';
    ctx.lineWidth = 1;
    
    // Horizontal lines
    for (let i = 0; i <= 4; i++) {
      const y = (height / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    
    // Vertical lines
    for (let i = 0; i <= 8; i++) {
      const x = (width / 8) * i;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    
    // Center line
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
  }
  
  drawAnalogSignal() {
    this.drawSignal(this.analogCanvas, (ctx, width, height) => {
      const duration = 2; // seconds
      const points = 1000;
      
      ctx.strokeStyle = '#6366f1';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      for (let i = 0; i < points; i++) {
        const t = (i / points) * duration;
        const x = (i / points) * width;
        const y = height / 2 - (this.generateSignal(t) * height * 0.4);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      
      ctx.stroke();
    });
  }
  
  drawSampledSignal() {
    this.drawSignal(this.sampledCanvas, (ctx, width, height) => {
      const duration = 2;
      const numSamples = Math.floor(this.sampleRate * duration);
      
      // Draw continuous signal faintly
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.2)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      
      for (let i = 0; i < 1000; i++) {
        const t = (i / 1000) * duration;
        const x = (i / 1000) * width;
        const y = height / 2 - (this.generateSignal(t) * height * 0.4);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      
      // Draw sample points
      ctx.fillStyle = '#8b5cf6';
      for (let i = 0; i < numSamples; i++) {
        const t = (i / this.sampleRate);
        const x = (t / duration) * width;
        const y = height / 2 - (this.generateSignal(t) * height * 0.4);
        
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw stem
        ctx.strokeStyle = '#8b5cf6';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, height / 2);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
    });
  }
  
  quantize(value) {
    const levels = Math.pow(2, this.bitDepth);
    const step = 2 / levels; // Range from -1 to 1
    return Math.round(value / step) * step;
  }
  
  drawQuantizedSignal() {
    this.drawSignal(this.quantizedCanvas, (ctx, width, height) => {
      const duration = 2;
      const numSamples = Math.floor(this.sampleRate * duration);
      
      // Draw quantization levels
      const levels = Math.pow(2, this.bitDepth);
      ctx.strokeStyle = 'rgba(236, 72, 153, 0.1)';
      ctx.lineWidth = 1;
      
      for (let i = 0; i <= levels; i++) {
        const value = -1 + (2 * i / levels);
        const y = height / 2 - (value * height * 0.4);
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
      
      // Draw quantized signal
      ctx.strokeStyle = '#ec4899';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      for (let i = 0; i < numSamples; i++) {
        const t = i / this.sampleRate;
        const x = (t / duration) * width;
        const value = this.generateSignal(t);
        const quantizedValue = this.quantize(value);
        const y = height / 2 - (quantizedValue * height * 0.4);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      
      // Draw sample points
      ctx.fillStyle = '#ec4899';
      for (let i = 0; i < numSamples; i++) {
        const t = i / this.sampleRate;
        const x = (t / duration) * width;
        const value = this.generateSignal(t);
        const quantizedValue = this.quantize(value);
        const y = height / 2 - (quantizedValue * height * 0.4);
        
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    });
  }
  
  drawReconstructedSignal() {
    this.drawSignal(this.reconstructedCanvas, (ctx, width, height) => {
      const duration = 2;
      const numSamples = Math.floor(this.sampleRate * duration);
      
      // Get quantized samples
      const samples = [];
      for (let i = 0; i < numSamples; i++) {
        const t = i / this.sampleRate;
        const value = this.generateSignal(t);
        samples.push(this.quantize(value));
      }
      
      // Draw original signal faintly
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.2)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      
      for (let i = 0; i < 1000; i++) {
        const t = (i / 1000) * duration;
        const x = (i / 1000) * width;
        const y = height / 2 - (this.generateSignal(t) * height * 0.4);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      
      // Draw reconstructed signal using linear interpolation
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      const reconstructionPoints = 1000;
      for (let i = 0; i < reconstructionPoints; i++) {
        const t = (i / reconstructionPoints) * duration;
        const sampleIndex = t * this.sampleRate;
        const index1 = Math.floor(sampleIndex);
        const index2 = Math.min(index1 + 1, numSamples - 1);
        const frac = sampleIndex - index1;
        
        const value = samples[index1] * (1 - frac) + samples[index2] * frac;
        const x = (i / reconstructionPoints) * width;
        const y = height / 2 - (value * height * 0.4);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    });
  }
  
  draw() {
    this.drawAnalogSignal();
    this.drawSampledSignal();
    this.drawQuantizedSignal();
    this.drawReconstructedSignal();
  }
  
  animate() {
    this.draw();
    requestAnimationFrame(() => this.animate());
  }
}

// ===========================
// AUDIO PREPROCESSING MODULE
// ===========================

class AudioPreprocessor {
  constructor() {
    this.audioContext = null;
    this.originalBuffer = null;
    this.processedBuffer = null;
    this.mediaRecorder = null;
    this.recordedChunks = [];
    this.isRecording = false;
    
    this.noiseThreshold = -40;
    this.silenceThreshold = -50;
    this.minSilenceDuration = 0.5;
    
    this.setupEventListeners();
  }
  
  setupEventListeners() {
    document.getElementById('audioFile').addEventListener('change', (e) => {
      this.loadAudioFile(e.target.files[0]);
    });
    
    document.getElementById('recordBtn').addEventListener('click', () => {
      this.toggleRecording();
    });
    
    document.getElementById('noiseThreshold').addEventListener('input', (e) => {
      this.noiseThreshold = parseFloat(e.target.value);
      document.getElementById('noiseThresholdValue').textContent = `${this.noiseThreshold} dB`;
    });
    
    document.getElementById('silenceThreshold').addEventListener('input', (e) => {
      this.silenceThreshold = parseFloat(e.target.value);
      document.getElementById('silenceThresholdValue').textContent = `${this.silenceThreshold} dB`;
    });
    
    document.getElementById('minSilenceDuration').addEventListener('input', (e) => {
      this.minSilenceDuration = parseFloat(e.target.value);
      document.getElementById('minSilenceDurationValue').textContent = `${this.minSilenceDuration} s`;
    });
    
    document.getElementById('applyNoiseReduction').addEventListener('click', () => {
      this.applyNoiseReduction();
    });
    
    document.getElementById('applySilenceRemoval').addEventListener('click', () => {
      this.applySilenceRemoval();
    });
    
    document.getElementById('resetAudio').addEventListener('click', () => {
      this.resetAudio();
    });
    
    document.getElementById('downloadAudio').addEventListener('click', () => {
      this.downloadProcessedAudio();
    });
  }
  
  async loadAudioFile(file) {
    if (!file) return;
    
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuffer = await file.arrayBuffer();
    this.originalBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
    
    this.processedBuffer = this.originalBuffer;
    this.displayAudio();
  }
  
  async toggleRecording() {
    if (!this.isRecording) {
      await this.startRecording();
    } else {
      this.stopRecording();
    }
  }
  
  async startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new MediaRecorder(stream);
      this.recordedChunks = [];
      
      this.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          this.recordedChunks.push(e.data);
        }
      };
      
      this.mediaRecorder.onstop = async () => {
        const blob = new Blob(this.recordedChunks, { type: 'audio/webm' });
        const arrayBuffer = await blob.arrayBuffer();
        
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.originalBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
        this.processedBuffer = this.originalBuffer;
        
        this.displayAudio();
        
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
      };
      
      this.mediaRecorder.start();
      this.isRecording = true;
      document.getElementById('recordBtnText').textContent = 'Stop Recording';
      document.getElementById('recordBtn').classList.add('pulse');
    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert('Could not access microphone. Please check permissions.');
    }
  }
  
  stopRecording() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();
      this.isRecording = false;
      document.getElementById('recordBtnText').textContent = 'Start Recording';
      document.getElementById('recordBtn').classList.remove('pulse');
    }
  }
  
  displayAudio() {
    document.getElementById('audioControls').style.display = 'block';
    
    this.drawWaveform(this.originalBuffer, document.getElementById('originalWaveform'));
    this.drawWaveform(this.processedBuffer, document.getElementById('processedWaveform'));
    
    this.setAudioSource(this.originalBuffer, document.getElementById('originalAudio'));
    this.setAudioSource(this.processedBuffer, document.getElementById('processedAudio'));
    
    this.updateDurationStats();
  }
  
  drawWaveform(audioBuffer, canvas) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.fillStyle = '#0a0e27';
    ctx.fillRect(0, 0, width, height);
    
    const data = audioBuffer.getChannelData(0);
    const step = Math.ceil(data.length / width);
    const amp = height / 2;
    
    // Draw waveform
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 1;
    ctx.beginPath();
    
    for (let i = 0; i < width; i++) {
      let min = 1.0;
      let max = -1.0;
      
      for (let j = 0; j < step; j++) {
        const datum = data[(i * step) + j];
        if (datum < min) min = datum;
        if (datum > max) max = datum;
      }
      
      const yMin = (1 + min) * amp;
      const yMax = (1 + max) * amp;
      
      if (i === 0) {
        ctx.moveTo(i, yMin);
      }
      
      ctx.lineTo(i, yMin);
      ctx.lineTo(i, yMax);
    }
    
    ctx.stroke();
    
    // Draw center line
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
  }
  
  setAudioSource(audioBuffer, audioElement) {
    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    
    const blob = this.bufferToWave(audioBuffer);
    const url = URL.createObjectURL(blob);
    audioElement.src = url;
  }
  
  bufferToWave(audioBuffer) {
    const numOfChan = audioBuffer.numberOfChannels;
    const length = audioBuffer.length * numOfChan * 2 + 44;
    const buffer = new ArrayBuffer(length);
    const view = new DataView(buffer);
    const channels = [];
    let offset = 0;
    let pos = 0;
    
    // Write WAV header
    const setUint16 = (data) => {
      view.setUint16(pos, data, true);
      pos += 2;
    };
    const setUint32 = (data) => {
      view.setUint32(pos, data, true);
      pos += 4;
    };
    
    setUint32(0x46464952); // "RIFF"
    setUint32(length - 8); // file length - 8
    setUint32(0x45564157); // "WAVE"
    setUint32(0x20746d66); // "fmt " chunk
    setUint32(16); // length = 16
    setUint16(1); // PCM (uncompressed)
    setUint16(numOfChan);
    setUint32(audioBuffer.sampleRate);
    setUint32(audioBuffer.sampleRate * 2 * numOfChan); // avg. bytes/sec
    setUint16(numOfChan * 2); // block-align
    setUint16(16); // 16-bit
    setUint32(0x61746164); // "data" - chunk
    setUint32(length - pos - 4); // chunk length
    
    // Write interleaved data
    for (let i = 0; i < audioBuffer.numberOfChannels; i++) {
      channels.push(audioBuffer.getChannelData(i));
    }
    
    while (pos < length) {
      for (let i = 0; i < numOfChan; i++) {
        let sample = Math.max(-1, Math.min(1, channels[i][offset]));
        sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        view.setInt16(pos, sample, true);
        pos += 2;
      }
      offset++;
    }
    
    return new Blob([buffer], { type: 'audio/wav' });
  }
  
  applyNoiseReduction() {
    if (!this.originalBuffer) return;
    
    const channelData = this.originalBuffer.getChannelData(0);
    const sampleRate = this.originalBuffer.sampleRate;
    const thresholdLinear = Math.pow(10, this.noiseThreshold / 20);
    
    // Simple noise gate
    const processedData = new Float32Array(channelData.length);
    for (let i = 0; i < channelData.length; i++) {
      const amplitude = Math.abs(channelData[i]);
      if (amplitude > thresholdLinear) {
        processedData[i] = channelData[i];
      } else {
        processedData[i] = channelData[i] * 0.1; // Reduce noise by 90%
      }
    }
    
    // Create new buffer
    this.processedBuffer = this.audioContext.createBuffer(
      this.originalBuffer.numberOfChannels,
      this.originalBuffer.length,
      sampleRate
    );
    this.processedBuffer.copyToChannel(processedData, 0);
    
    this.drawWaveform(this.processedBuffer, document.getElementById('processedWaveform'));
    this.setAudioSource(this.processedBuffer, document.getElementById('processedAudio'));
    this.updateDurationStats();
  }
  
  applySilenceRemoval() {
    if (!this.originalBuffer) return;
    
    const channelData = this.originalBuffer.getChannelData(0);
    const sampleRate = this.originalBuffer.sampleRate;
    const thresholdLinear = Math.pow(10, this.silenceThreshold / 20);
    const minSilenceSamples = Math.floor(this.minSilenceDuration * sampleRate);
    
    // Detect non-silent regions
    const segments = [];
    let inSilence = true;
    let silenceStart = 0;
    let segmentStart = 0;
    
    for (let i = 0; i < channelData.length; i++) {
      const amplitude = Math.abs(channelData[i]);
      
      if (amplitude > thresholdLinear) {
        if (inSilence) {
          const silenceDuration = i - silenceStart;
          if (silenceDuration >= minSilenceSamples && segments.length > 0) {
            // End previous segment
            segments[segments.length - 1].end = silenceStart;
          }
          segmentStart = i;
          inSilence = false;
        }
        if (segments.length === 0 || segments[segments.length - 1].end !== undefined) {
          segments.push({ start: segmentStart, end: undefined });
        }
      } else {
        if (!inSilence) {
          silenceStart = i;
          inSilence = true;
        }
      }
    }
    
    // Close last segment
    if (segments.length > 0 && segments[segments.length - 1].end === undefined) {
      segments[segments.length - 1].end = channelData.length;
    }
    
    // Concatenate non-silent segments
    const totalLength = segments.reduce((sum, seg) => sum + (seg.end - seg.start), 0);
    const processedData = new Float32Array(totalLength);
    
    let offset = 0;
    for (const segment of segments) {
      const segmentData = channelData.slice(segment.start, segment.end);
      processedData.set(segmentData, offset);
      offset += segmentData.length;
    }
    
    // Create new buffer
    this.processedBuffer = this.audioContext.createBuffer(
      this.originalBuffer.numberOfChannels,
      totalLength,
      sampleRate
    );
    this.processedBuffer.copyToChannel(processedData, 0);
    
    this.drawWaveform(this.processedBuffer, document.getElementById('processedWaveform'));
    this.setAudioSource(this.processedBuffer, document.getElementById('processedAudio'));
    this.updateDurationStats();
  }
  
  resetAudio() {
    if (this.originalBuffer) {
      this.processedBuffer = this.originalBuffer;
      this.drawWaveform(this.processedBuffer, document.getElementById('processedWaveform'));
      this.setAudioSource(this.processedBuffer, document.getElementById('processedAudio'));
      this.updateDurationStats();
    }
  }
  
  updateDurationStats() {
    if (!this.originalBuffer || !this.processedBuffer) return;
    
    const originalDuration = this.originalBuffer.duration;
    const processedDuration = this.processedBuffer.duration;
    const reduction = ((originalDuration - processedDuration) / originalDuration * 100).toFixed(1);
    
    document.getElementById('originalDuration').textContent = `${originalDuration.toFixed(2)}s`;
    document.getElementById('processedDuration').textContent = `${processedDuration.toFixed(2)}s`;
    document.getElementById('reductionPercent').textContent = `${reduction}%`;
  }
  
  downloadProcessedAudio() {
    if (!this.processedBuffer) return;
    
    const blob = this.bufferToWave(this.processedBuffer);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'processed_audio.wav';
    a.click();
    URL.revokeObjectURL(url);
  }
}

// ===========================
// INITIALIZATION
// ===========================

document.addEventListener('DOMContentLoaded', () => {
  const adcSimulator = new ADCSimulator();
  const audioPreprocessor = new AudioPreprocessor();
  
  // Initialize stats
  adcSimulator.updateStats();
});
