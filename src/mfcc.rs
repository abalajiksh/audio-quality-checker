// src/mfcc.rs
//
// Mel-Frequency Cepstral Coefficients (MFCC) analysis.
// Useful for codec signature detection and audio fingerprinting.

use crate::decoder::AudioData;
use crate::dsp::{FftProcessor, WindowType, amplitude_to_db};
use std::f32::consts::PI;

/// MFCC analysis result
#[derive(Debug, Clone)]
pub struct MfccAnalysis {
    /// Number of MFCC coefficients computed
    pub num_coefficients: usize,
    /// Average MFCC values across the track
    pub mean_mfcc: Vec<f32>,
    /// Standard deviation of MFCCs
    pub std_mfcc: Vec<f32>,
    /// Delta MFCCs (first derivative)
    pub mean_delta_mfcc: Vec<f32>,
    /// MFCC-based similarity score to known codec signatures
    pub codec_similarity_scores: Vec<(String, f32)>,
    /// Evidence strings
    pub evidence: Vec<String>,
}

/// Parameters for MFCC computation
pub struct MfccParams {
    /// Number of MFCC coefficients to compute
    pub num_coeffs: usize,
    /// Number of mel filter banks
    pub num_mel_filters: usize,
    /// FFT size
    pub fft_size: usize,
    /// Hop size
    pub hop_size: usize,
    /// Minimum frequency for mel filters
    pub min_freq: f32,
    /// Maximum frequency for mel filters (None = Nyquist)
    pub max_freq: Option<f32>,
}

impl Default for MfccParams {
    fn default() -> Self {
        MfccParams {
            num_coeffs: 13,
            num_mel_filters: 26,
            fft_size: 2048,
            hop_size: 512,
            min_freq: 20.0,
            max_freq: None,
        }
    }
}

/// Compute MFCCs for audio
pub fn analyze_mfcc(audio: &AudioData) -> MfccAnalysis {
    let params = MfccParams::default();
    let mono = crate::decoder::extract_mono(audio);
    
    let max_freq = params.max_freq.unwrap_or(audio.sample_rate as f32 / 2.0);
    
    // Create mel filterbank
    let mel_filters = create_mel_filterbank(
        params.num_mel_filters,
        params.fft_size / 2,
        audio.sample_rate,
        params.min_freq,
        max_freq,
    );
    
    // Create DCT matrix
    let dct_matrix = create_dct_matrix(params.num_coeffs, params.num_mel_filters);
    
    // Compute MFCCs for each frame
    let mut fft = FftProcessor::new(params.fft_size, WindowType::Hann);
    let num_frames = (mono.len().saturating_sub(params.fft_size)) / params.hop_size;
    
    if num_frames == 0 {
        return MfccAnalysis {
            num_coefficients: params.num_coeffs,
            mean_mfcc: vec![0.0; params.num_coeffs],
            std_mfcc: vec![0.0; params.num_coeffs],
            mean_delta_mfcc: vec![0.0; params.num_coeffs],
            codec_similarity_scores: vec![],
            evidence: vec!["Audio too short for MFCC analysis".to_string()],
        };
    }
    
    let mut all_mfccs: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
    
    for frame_idx in 0..num_frames {
        let start = frame_idx * params.hop_size;
        if start + params.fft_size > mono.len() {
            break;
        }
        
        let frame = &mono[start..start + params.fft_size];
        let power_spectrum = fft.magnitude_spectrum(frame);
        
        // Apply mel filterbank
        let mel_energies = apply_mel_filterbank(&power_spectrum, &mel_filters);
        
        // Log compression
        let log_mel: Vec<f32> = mel_energies.iter()
            .map(|&e| (e + 1e-10).ln())
            .collect();
        
        // DCT to get MFCCs
        let mfcc = apply_dct(&log_mel, &dct_matrix);
        all_mfccs.push(mfcc);
    }
    
    // Calculate statistics
    let mean_mfcc = calculate_mean_mfcc(&all_mfccs, params.num_coeffs);
    let std_mfcc = calculate_std_mfcc(&all_mfccs, &mean_mfcc);
    let mean_delta_mfcc = calculate_delta_mfcc(&all_mfccs, params.num_coeffs);
    
    // Compare to known codec signatures
    let codec_similarity_scores = compare_to_codec_signatures(&mean_mfcc, &std_mfcc);
    
    let mut evidence = Vec::new();
    evidence.push(format!("Computed {} MFCCs over {} frames", params.num_coeffs, all_mfccs.len()));
    
    for (codec, score) in &codec_similarity_scores {
        if *score > 0.7 {
            evidence.push(format!("High similarity to {}: {:.1}%", codec, score * 100.0));
        }
    }
    
    MfccAnalysis {
        num_coefficients: params.num_coeffs,
        mean_mfcc,
        std_mfcc,
        mean_delta_mfcc,
        codec_similarity_scores,
        evidence,
    }
}

/// Create mel filterbank
fn create_mel_filterbank(
    num_filters: usize,
    num_fft_bins: usize,
    sample_rate: u32,
    min_freq: f32,
    max_freq: f32,
) -> Vec<Vec<f32>> {
    let min_mel = hz_to_mel(min_freq);
    let max_mel = hz_to_mel(max_freq);
    
    // Mel points evenly spaced
    let mel_points: Vec<f32> = (0..=num_filters + 1)
        .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (num_filters + 1) as f32)
        .collect();
    
    // Convert back to Hz
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    
    // Convert to FFT bin indices
    let bin_points: Vec<usize> = hz_points.iter()
        .map(|&f| ((f / (sample_rate as f32 / 2.0)) * num_fft_bins as f32) as usize)
        .map(|b| b.min(num_fft_bins - 1))
        .collect();
    
    // Create triangular filters
    let mut filters = vec![vec![0.0f32; num_fft_bins]; num_filters];
    
    for i in 0..num_filters {
        let left = bin_points[i];
        let center = bin_points[i + 1];
        let right = bin_points[i + 2];
        
        // Rising edge
        for j in left..center {
            if center > left {
                filters[i][j] = (j - left) as f32 / (center - left) as f32;
            }
        }
        
        // Falling edge
        for j in center..=right {
            if right > center {
                filters[i][j] = (right - j) as f32 / (right - center) as f32;
            }
        }
    }
    
    filters
}

/// Create DCT-II matrix for MFCC computation
fn create_dct_matrix(num_coeffs: usize, num_filters: usize) -> Vec<Vec<f32>> {
    let mut dct = vec![vec![0.0f32; num_filters]; num_coeffs];
    
    for k in 0..num_coeffs {
        for n in 0..num_filters {
            dct[k][n] = (PI * k as f32 * (2.0 * n as f32 + 1.0) / (2.0 * num_filters as f32)).cos();
        }
    }
    
    // Normalize
    let norm = (2.0 / num_filters as f32).sqrt();
    for row in &mut dct {
        for val in row {
            *val *= norm;
        }
    }
    
    dct
}

/// Apply mel filterbank to power spectrum
fn apply_mel_filterbank(spectrum: &[f32], filters: &[Vec<f32>]) -> Vec<f32> {
    filters.iter()
        .map(|filter| {
            spectrum.iter()
                .zip(filter)
                .map(|(&s, &f)| s * s * f)  // Power spectrum
                .sum()
        })
        .collect()
}

/// Apply DCT to get MFCCs
fn apply_dct(log_mel: &[f32], dct_matrix: &[Vec<f32>]) -> Vec<f32> {
    dct_matrix.iter()
        .map(|row| {
            row.iter()
                .zip(log_mel)
                .map(|(&d, &l)| d * l)
                .sum()
        })
        .collect()
}

/// Calculate mean MFCC across frames
fn calculate_mean_mfcc(all_mfccs: &[Vec<f32>], num_coeffs: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; num_coeffs];
    
    for mfcc in all_mfccs {
        for (i, &val) in mfcc.iter().enumerate() {
            if i < num_coeffs {
                mean[i] += val;
            }
        }
    }
    
    let n = all_mfccs.len() as f32;
    for val in &mut mean {
        *val /= n;
    }
    
    mean
}

/// Calculate standard deviation of MFCCs
fn calculate_std_mfcc(all_mfccs: &[Vec<f32>], mean: &[f32]) -> Vec<f32> {
    let num_coeffs = mean.len();
    let mut variance = vec![0.0f32; num_coeffs];
    
    for mfcc in all_mfccs {
        for (i, &val) in mfcc.iter().enumerate() {
            if i < num_coeffs {
                let diff = val - mean[i];
                variance[i] += diff * diff;
            }
        }
    }
    
    let n = all_mfccs.len() as f32;
    variance.iter().map(|&v| (v / n).sqrt()).collect()
}

/// Calculate delta MFCCs (first derivative)
fn calculate_delta_mfcc(all_mfccs: &[Vec<f32>], num_coeffs: usize) -> Vec<f32> {
    if all_mfccs.len() < 3 {
        return vec![0.0; num_coeffs];
    }
    
    let mut delta_sum = vec![0.0f32; num_coeffs];
    let window = 2;
    
    for i in window..all_mfccs.len() - window {
        for coeff in 0..num_coeffs.min(all_mfccs[i].len()) {
            let mut num = 0.0f32;
            let mut denom = 0.0f32;
            
            for j in 1..=window {
                let j_f = j as f32;
                if i + j < all_mfccs.len() && i >= j && coeff < all_mfccs[i + j].len() && coeff < all_mfccs[i - j].len() {
                    num += j_f * (all_mfccs[i + j][coeff] - all_mfccs[i - j][coeff]);
                    denom += j_f * j_f;
                }
            }
            
            if denom > 0.0 {
                delta_sum[coeff] += num / (2.0 * denom);
            }
        }
    }
    
    let n = (all_mfccs.len() - 2 * window) as f32;
    delta_sum.iter().map(|&d| d / n.max(1.0)).collect()
}

/// Compare MFCCs to known codec signatures
fn compare_to_codec_signatures(mean: &[f32], std: &[f32]) -> Vec<(String, f32)> {
    // These are example reference signatures - in a real implementation,
    // you would build these from a training set of known transcoded files
    let codec_signatures = get_reference_signatures();
    
    let mut scores = Vec::new();
    
    for (name, ref_mean, ref_std) in codec_signatures {
        let similarity = calculate_mfcc_similarity(mean, std, &ref_mean, &ref_std);
        scores.push((name.to_string(), similarity));
    }
    
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores
}

/// Reference MFCC signatures for common codecs
fn get_reference_signatures() -> Vec<(&'static str, Vec<f32>, Vec<f32>)> {
    // These are placeholder values - real signatures would be computed from
    // analyzing many known transcoded files
    vec![
        ("MP3 128kbps", vec![-2.0, 1.0, -0.5, 0.3, -0.2, 0.1, 0.0, 0.1, -0.1, 0.05, 0.0, 0.0, 0.0],
                        vec![0.8, 0.6, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.15, 0.1, 0.1, 0.1]),
        ("AAC 128kbps", vec![-1.8, 1.2, -0.4, 0.4, -0.3, 0.15, 0.05, 0.08, -0.05, 0.03, 0.0, 0.0, 0.0],
                        vec![0.7, 0.55, 0.45, 0.35, 0.28, 0.25, 0.18, 0.18, 0.15, 0.12, 0.08, 0.08, 0.08]),
        ("Vorbis q5", vec![-1.9, 1.1, -0.45, 0.35, -0.25, 0.12, 0.02, 0.09, -0.08, 0.04, 0.0, 0.0, 0.0],
                      vec![0.75, 0.58, 0.48, 0.38, 0.29, 0.27, 0.19, 0.19, 0.17, 0.13, 0.09, 0.09, 0.09]),
        ("Lossless", vec![-1.5, 1.4, -0.3, 0.5, -0.35, 0.2, 0.1, 0.12, -0.02, 0.06, 0.02, 0.01, 0.01],
                     vec![0.9, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.22, 0.18, 0.14, 0.12, 0.1]),
    ]
}

/// Calculate similarity between MFCC signatures
fn calculate_mfcc_similarity(
    mean1: &[f32], std1: &[f32],
    mean2: &[f32], std2: &[f32],
) -> f32 {
    let num_coeffs = mean1.len().min(mean2.len()).min(std1.len()).min(std2.len());
    
    if num_coeffs == 0 {
        return 0.0;
    }
    
    // Mahalanobis-like distance
    let mut distance = 0.0f32;
    
    for i in 0..num_coeffs {
        let combined_var = std1[i] * std1[i] + std2[i] * std2[i] + 0.01;  // Add small constant
        let diff = mean1[i] - mean2[i];
        distance += diff * diff / combined_var;
    }
    
    // Convert distance to similarity (0 to 1)
    let avg_distance = distance / num_coeffs as f32;
    (1.0 / (1.0 + avg_distance)).sqrt()
}

/// Convert Hz to mel scale
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel to Hz
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hz_mel_conversion() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let hz_back = mel_to_hz(mel);
        assert!((hz - hz_back).abs() < 0.1);
    }

    #[test]
    fn test_dct_matrix_creation() {
        let dct = create_dct_matrix(13, 26);
        assert_eq!(dct.len(), 13);
        assert_eq!(dct[0].len(), 26);
    }
}
