// src/dsp.rs
//
// Core Digital Signal Processing utilities.
// Pure math implementations - no ML/AI.

use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

/// FFT computation with windowing
pub struct FftProcessor {
    planner: FftPlanner<f32>,
    window: Vec<f32>,
    fft_size: usize,
}

impl FftProcessor {
    pub fn new(fft_size: usize, window_type: WindowType) -> Self {
        let window = create_window(fft_size, window_type);
        Self {
            planner: FftPlanner::new(),
            window,
            fft_size,
        }
    }

    /// Compute magnitude spectrum
    pub fn magnitude_spectrum(&mut self, samples: &[f32]) -> Vec<f32> {
        let fft = self.planner.plan_fft_forward(self.fft_size);
        
        let mut buffer: Vec<Complex<f32>> = samples
            .iter()
            .take(self.fft_size)
            .enumerate()
            .map(|(i, &s)| Complex::new(s * self.window[i], 0.0))
            .collect();
        
        // Zero-pad if necessary
        buffer.resize(self.fft_size, Complex::new(0.0, 0.0));
        
        fft.process(&mut buffer);
        
        buffer[..self.fft_size / 2]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect()
    }

    /// Compute power spectrum in dB
    pub fn power_spectrum_db(&mut self, samples: &[f32]) -> Vec<f32> {
        let mags = self.magnitude_spectrum(samples);
        mags.iter()
            .map(|&m| {
                if m > 1e-10 {
                    20.0 * m.log10()
                } else {
                    -200.0
                }
            })
            .collect()
    }

    /// Compute complex spectrum (for phase analysis)
    pub fn complex_spectrum(&mut self, samples: &[f32]) -> Vec<Complex<f32>> {
        let fft = self.planner.plan_fft_forward(self.fft_size);
        
        let mut buffer: Vec<Complex<f32>> = samples
            .iter()
            .take(self.fft_size)
            .enumerate()
            .map(|(i, &s)| Complex::new(s * self.window[i], 0.0))
            .collect();
        
        buffer.resize(self.fft_size, Complex::new(0.0, 0.0));
        fft.process(&mut buffer);
        
        buffer[..self.fft_size / 2].to_vec()
    }

    pub fn fft_size(&self) -> usize {
        self.fft_size
    }
}

/// Window function types
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    Hann,
    Hamming,
    Blackman,
    BlackmanHarris,
    FlatTop,
    Kaiser(f32),  // Beta parameter
}

/// Create window function
pub fn create_window(size: usize, window_type: WindowType) -> Vec<f32> {
    let n = size as f32;
    (0..size)
        .map(|i| {
            let x = i as f32;
            match window_type {
                WindowType::Hann => {
                    0.5 * (1.0 - (2.0 * PI * x / n).cos())
                }
                WindowType::Hamming => {
                    0.54 - 0.46 * (2.0 * PI * x / n).cos()
                }
                WindowType::Blackman => {
                    0.42 - 0.5 * (2.0 * PI * x / n).cos() 
                        + 0.08 * (4.0 * PI * x / n).cos()
                }
                WindowType::BlackmanHarris => {
                    0.35875 - 0.48829 * (2.0 * PI * x / n).cos()
                        + 0.14128 * (4.0 * PI * x / n).cos()
                        - 0.01168 * (6.0 * PI * x / n).cos()
                }
                WindowType::FlatTop => {
                    // Good for amplitude accuracy
                    0.21557895 - 0.41663158 * (2.0 * PI * x / n).cos()
                        + 0.277263158 * (4.0 * PI * x / n).cos()
                        - 0.083578947 * (6.0 * PI * x / n).cos()
                        + 0.006947368 * (8.0 * PI * x / n).cos()
                }
                WindowType::Kaiser(beta) => {
                    let alpha = (n - 1.0) / 2.0;
                    let ratio = (x - alpha) / alpha;
                    let arg = beta * (1.0 - ratio * ratio).max(0.0).sqrt();
                    bessel_i0(arg) / bessel_i0(beta)
                }
            }
        })
        .collect()
}

/// Modified Bessel function I0 (for Kaiser window)
fn bessel_i0(x: f32) -> f32 {
    let mut sum = 1.0f32;
    let mut term = 1.0f32;
    let x2 = x * x;
    
    for k in 1..50 {
        term *= x2 / (4.0 * k as f32 * k as f32);
        sum += term;
        if term < 1e-10 {
            break;
        }
    }
    sum
}

/// Apply pre-emphasis filter (boosts high frequencies)
pub fn pre_emphasis(samples: &[f32], coefficient: f32) -> Vec<f32> {
    if samples.is_empty() {
        return vec![];
    }
    
    let mut output = Vec::with_capacity(samples.len());
    output.push(samples[0]);
    
    for i in 1..samples.len() {
        output.push(samples[i] - coefficient * samples[i - 1]);
    }
    
    output
}

/// Apply de-emphasis filter (inverse of pre-emphasis)
pub fn de_emphasis(samples: &[f32], coefficient: f32) -> Vec<f32> {
    if samples.is_empty() {
        return vec![];
    }
    
    let mut output = Vec::with_capacity(samples.len());
    output.push(samples[0]);
    
    for i in 1..samples.len() {
        output.push(samples[i] + coefficient * output[i - 1]);
    }
    
    output
}

/// Compute moving average
pub fn moving_average(data: &[f32], window_size: usize) -> Vec<f32> {
    if data.len() < window_size || window_size == 0 {
        return data.to_vec();
    }
    
    let mut result = Vec::with_capacity(data.len());
    let mut sum: f32 = data[..window_size].iter().sum();
    
    // First window_size/2 values: use partial window
    for i in 0..window_size / 2 {
        let partial_sum: f32 = data[..=i + window_size / 2].iter().sum();
        result.push(partial_sum / (i + window_size / 2 + 1) as f32);
    }
    
    // Middle values: full window
    for i in window_size / 2..data.len() - window_size / 2 {
        if i > window_size / 2 {
            sum = sum - data[i - window_size / 2 - 1] + data[i + window_size / 2];
        }
        result.push(sum / window_size as f32);
    }
    
    // Last window_size/2 values: use partial window
    for i in data.len() - window_size / 2..data.len() {
        let partial_sum: f32 = data[i - window_size / 2..].iter().sum();
        result.push(partial_sum / (data.len() - i + window_size / 2) as f32);
    }
    
    // Ensure output length matches input
    result.truncate(data.len());
    while result.len() < data.len() {
        result.push(*data.last().unwrap_or(&0.0));
    }
    
    result
}

/// Compute median of a slice
pub fn median(data: &mut [f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let mid = data.len() / 2;
    if data.len() % 2 == 0 {
        (data[mid - 1] + data[mid]) / 2.0
    } else {
        data[mid]
    }
}

/// Compute RMS (Root Mean Square)
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Compute peak amplitude
pub fn peak_amplitude(samples: &[f32]) -> f32 {
    samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
}

/// Convert amplitude to dB (relative to 1.0)
pub fn amplitude_to_db(amplitude: f32) -> f32 {
    if amplitude > 1e-10 {
        20.0 * amplitude.log10()
    } else {
        -200.0
    }
}

/// Convert dB to amplitude
pub fn db_to_amplitude(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Compute envelope using Hilbert transform approximation
pub fn compute_envelope(samples: &[f32], smooth_samples: usize) -> Vec<f32> {
    // Simple peak-following envelope
    let mut envelope = Vec::with_capacity(samples.len());
    
    let attack = 0.01;  // Fast attack
    let release = 0.0001;  // Slow release
    
    let mut current = 0.0f32;
    
    for &sample in samples {
        let abs_sample = sample.abs();
        if abs_sample > current {
            current = current + attack * (abs_sample - current);
        } else {
            current = current + release * (abs_sample - current);
        }
        envelope.push(current);
    }
    
    // Smooth the envelope
    if smooth_samples > 0 {
        moving_average(&envelope, smooth_samples)
    } else {
        envelope
    }
}

/// Find transient positions (sudden amplitude increases)
pub fn find_transients(samples: &[f32], threshold_db: f32, min_distance: usize) -> Vec<usize> {
    let envelope = compute_envelope(samples, 64);
    let envelope_db: Vec<f32> = envelope.iter()
        .map(|&e| amplitude_to_db(e))
        .collect();
    
    let mut transients = Vec::new();
    let mut last_transient = 0;
    
    // Look for sudden increases in envelope
    let analysis_hop = 32;
    for i in (analysis_hop..envelope_db.len() - analysis_hop).step_by(analysis_hop) {
        let before = envelope_db[i - analysis_hop..i].iter()
            .fold(f32::MIN, |a, &b| a.max(b));
        let after = envelope_db[i..i + analysis_hop].iter()
            .fold(f32::MIN, |a, &b| a.max(b));
        
        let increase = after - before;
        
        if increase > threshold_db && i - last_transient > min_distance {
            transients.push(i);
            last_transient = i;
        }
    }
    
    transients
}

/// Simple sinc interpolation for upsampling
pub fn upsample_sinc(samples: &[f32], factor: usize) -> Vec<f32> {
    if factor <= 1 {
        return samples.to_vec();
    }
    
    let output_len = samples.len() * factor;
    let mut output = vec![0.0f32; output_len];
    
    // Sinc filter parameters
    let filter_len = 32;
    
    for (i, &sample) in samples.iter().enumerate() {
        output[i * factor] = sample;
    }
    
    // Interpolate intermediate samples
    for i in 0..output_len {
        if i % factor == 0 {
            continue;  // Original sample
        }
        
        let fractional_pos = i as f32 / factor as f32;
        let base_idx = fractional_pos.floor() as i32;
        let frac = fractional_pos - base_idx as f32;
        
        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;
        
        for j in -filter_len..=filter_len {
            let idx = base_idx + j;
            if idx >= 0 && (idx as usize) < samples.len() {
                let x = (j as f32 - frac) * PI;
                let sinc = if x.abs() < 1e-6 { 1.0 } else { x.sin() / x };
                
                // Apply window
                let window_x = (j as f32 - frac) / filter_len as f32;
                let window = if window_x.abs() <= 1.0 {
                    0.5 * (1.0 + (PI * window_x).cos())
                } else {
                    0.0
                };
                
                let weight = sinc * window;
                sum += samples[idx as usize] * weight;
                weight_sum += weight.abs();
            }
        }
        
        output[i] = if weight_sum > 0.0 { sum / weight_sum * factor as f32 } else { 0.0 };
    }
    
    output
}

/// Simple downsampling with anti-aliasing
pub fn downsample_simple(samples: &[f32], factor: usize) -> Vec<f32> {
    if factor <= 1 {
        return samples.to_vec();
    }
    
    // Apply simple averaging as anti-aliasing
    samples.chunks(factor)
        .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
        .collect()
}

/// Zero-crossing rate
pub fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    
    let crossings: usize = samples.windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    
    crossings as f32 / (samples.len() - 1) as f32
}

/// Compute autocorrelation
pub fn autocorrelation(samples: &[f32], max_lag: usize) -> Vec<f32> {
    let n = samples.len();
    let max_lag = max_lag.min(n - 1);
    
    // Normalize by energy
    let energy: f32 = samples.iter().map(|s| s * s).sum();
    if energy < 1e-10 {
        return vec![0.0; max_lag + 1];
    }
    
    (0..=max_lag)
        .map(|lag| {
            let sum: f32 = samples[..n - lag].iter()
                .zip(&samples[lag..])
                .map(|(a, b)| a * b)
                .sum();
            sum / energy
        })
        .collect()
}

/// Compute spectral centroid (brightness measure)
pub fn spectral_centroid(magnitudes: &[f32], sample_rate: u32) -> f32 {
    let total_energy: f32 = magnitudes.iter().sum();
    if total_energy < 1e-10 {
        return 0.0;
    }
    
    let weighted_sum: f32 = magnitudes.iter()
        .enumerate()
        .map(|(i, &m)| {
            let freq = i as f32 * sample_rate as f32 / (2.0 * magnitudes.len() as f32);
            freq * m
        })
        .sum();
    
    weighted_sum / total_energy
}

/// Compute spectral spread (bandwidth)
pub fn spectral_spread(magnitudes: &[f32], sample_rate: u32) -> f32 {
    let centroid = spectral_centroid(magnitudes, sample_rate);
    let total_energy: f32 = magnitudes.iter().sum();
    
    if total_energy < 1e-10 {
        return 0.0;
    }
    
    let variance: f32 = magnitudes.iter()
        .enumerate()
        .map(|(i, &m)| {
            let freq = i as f32 * sample_rate as f32 / (2.0 * magnitudes.len() as f32);
            let diff = freq - centroid;
            diff * diff * m
        })
        .sum();
    
    (variance / total_energy).sqrt()
}

/// Compute spectral flatness (Wiener entropy)
/// Returns 1.0 for white noise, approaches 0.0 for tonal signals
pub fn spectral_flatness(magnitudes: &[f32]) -> f32 {
    let n = magnitudes.len() as f32;
    
    // Geometric mean (via log)
    let log_sum: f32 = magnitudes.iter()
        .map(|&m| (m + 1e-10).ln())
        .sum();
    let geometric_mean = (log_sum / n).exp();
    
    // Arithmetic mean
    let arithmetic_mean = magnitudes.iter().sum::<f32>() / n;
    
    if arithmetic_mean < 1e-10 {
        return 0.0;
    }
    
    geometric_mean / arithmetic_mean
}

/// Compute spectral rolloff (frequency below which X% of energy is contained)
pub fn spectral_rolloff(magnitudes: &[f32], sample_rate: u32, percentile: f32) -> f32 {
    let total_energy: f32 = magnitudes.iter().map(|m| m * m).sum();
    let threshold = total_energy * percentile;
    
    let mut cumulative = 0.0f32;
    
    for (i, &mag) in magnitudes.iter().enumerate() {
        cumulative += mag * mag;
        if cumulative >= threshold {
            return i as f32 * sample_rate as f32 / (2.0 * magnitudes.len() as f32);
        }
    }
    
    sample_rate as f32 / 2.0
}

/// Compute spectral flux (frame-to-frame spectral change)
pub fn spectral_flux(prev_spectrum: &[f32], curr_spectrum: &[f32]) -> f32 {
    if prev_spectrum.len() != curr_spectrum.len() {
        return 0.0;
    }
    
    // Rectified spectral flux (only positive changes)
    prev_spectrum.iter()
        .zip(curr_spectrum)
        .map(|(&prev, &curr)| {
            let diff = curr - prev;
            if diff > 0.0 { diff * diff } else { 0.0 }
        })
        .sum::<f32>()
        .sqrt()
}

/// Compute spectral contrast in frequency bands
pub fn spectral_contrast(magnitudes: &[f32], num_bands: usize) -> Vec<f32> {
    let band_size = magnitudes.len() / num_bands;
    
    (0..num_bands)
        .map(|band| {
            let start = band * band_size;
            let end = ((band + 1) * band_size).min(magnitudes.len());
            let band_mags = &magnitudes[start..end];
            
            if band_mags.is_empty() {
                return 0.0;
            }
            
            let mut sorted: Vec<f32> = band_mags.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
            // Top 20% (peaks) vs bottom 20% (valleys)
            let n = sorted.len();
            let top_start = (n * 80) / 100;
            let bottom_end = (n * 20) / 100;
            
            let peaks: f32 = sorted[top_start..].iter().sum::<f32>() / (n - top_start) as f32;
            let valleys: f32 = sorted[..bottom_end.max(1)].iter().sum::<f32>() / bottom_end.max(1) as f32;
            
            if valleys > 1e-10 {
                amplitude_to_db(peaks) - amplitude_to_db(valleys)
            } else {
                amplitude_to_db(peaks) + 60.0  // Large contrast
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let window = create_window(4, WindowType::Hann);
        assert!((window[0]).abs() < 0.01);  // Should be ~0 at edges
        assert!((window[2] - 1.0).abs() < 0.01);  // Should be ~1 at center
    }

    #[test]
    fn test_rms() {
        let samples = vec![1.0, -1.0, 1.0, -1.0];
        assert!((rms(&samples) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_spectral_flatness_tonal() {
        // Mostly zeros with one peak = low flatness
        let mut mags = vec![0.001; 100];
        mags[50] = 1.0;
        let flatness = spectral_flatness(&mags);
        assert!(flatness < 0.1);
    }

    #[test]
    fn test_spectral_flatness_noise() {
        // All equal = high flatness
        let mags = vec![1.0; 100];
        let flatness = spectral_flatness(&mags);
        assert!(flatness > 0.99);
    }
}
