// src/stereo.rs
//
// Stereo field analysis for detecting joint stereo encoding
// and analyzing stereo image characteristics.

use crate::decoder::{AudioData, extract_stereo, compute_mid_side};
use crate::dsp::{FftProcessor, WindowType, rms};

/// Stereo analysis result
#[derive(Debug, Clone)]
pub struct StereoAnalysis {
    /// Whether the file is stereo
    pub is_stereo: bool,
    /// Overall stereo width (0.0 = mono, 1.0 = full stereo)
    pub stereo_width: f32,
    /// Correlation between L and R channels (-1.0 to 1.0)
    pub channel_correlation: f32,
    /// Whether joint stereo encoding was likely used
    pub joint_stereo_detected: bool,
    /// Confidence in joint stereo detection
    pub joint_stereo_confidence: f32,
    /// High frequency stereo reduction (characteristic of joint stereo)
    pub hf_stereo_reduction: f32,
    /// Mid/Side energy ratio
    pub mid_side_ratio: f32,
    /// Frequency above which stereo narrows (if joint stereo)
    pub stereo_narrowing_freq: Option<f32>,
    /// Evidence for the analysis
    pub evidence: Vec<String>,
}

/// Analyze stereo characteristics
pub fn analyze_stereo(audio: &AudioData) -> StereoAnalysis {
    if audio.channels < 2 {
        return StereoAnalysis {
            is_stereo: false,
            stereo_width: 0.0,
            channel_correlation: 1.0,
            joint_stereo_detected: false,
            joint_stereo_confidence: 0.0,
            hf_stereo_reduction: 0.0,
            mid_side_ratio: f32::INFINITY,
            stereo_narrowing_freq: None,
            evidence: vec!["Mono file".to_string()],
        };
    }
    
    let (left, right) = match extract_stereo(audio) {
        Some(channels) => channels,
        None => return StereoAnalysis::mono_default(),
    };
    
    let (mid, side) = match compute_mid_side(audio) {
        Some(ms) => ms,
        None => return StereoAnalysis::mono_default(),
    };
    
    let mut evidence = Vec::new();
    
    // Basic stereo width and correlation
    let stereo_width = calculate_stereo_width(&left, &right);
    let channel_correlation = calculate_correlation(&left, &right);
    
    evidence.push(format!("Stereo width: {:.1}%", stereo_width * 100.0));
    evidence.push(format!("L/R correlation: {:.3}", channel_correlation));
    
    // Mid/Side energy ratio
    let mid_energy = rms(&mid);
    let side_energy = rms(&side);
    let mid_side_ratio = if side_energy > 1e-10 {
        mid_energy / side_energy
    } else {
        f32::INFINITY
    };
    
    evidence.push(format!("M/S energy ratio: {:.2}", mid_side_ratio));
    
    // Frequency-dependent stereo analysis
    let (hf_stereo_reduction, narrowing_freq) = analyze_frequency_dependent_stereo(
        &left, &right, &mid, &side, audio.sample_rate
    );
    
    if hf_stereo_reduction > 0.3 {
        evidence.push(format!("HF stereo reduction: {:.1}%", hf_stereo_reduction * 100.0));
    }
    
    if let Some(freq) = narrowing_freq {
        evidence.push(format!("Stereo narrowing above: {:.0} Hz", freq));
    }
    
    // Detect joint stereo encoding
    let (joint_stereo_detected, joint_stereo_confidence) = detect_joint_stereo(
        stereo_width,
        channel_correlation,
        hf_stereo_reduction,
        mid_side_ratio,
        narrowing_freq,
    );
    
    if joint_stereo_detected {
        evidence.push(format!(
            "Joint stereo encoding likely (confidence: {:.1}%)",
            joint_stereo_confidence * 100.0
        ));
    }
    
    StereoAnalysis {
        is_stereo: true,
        stereo_width,
        channel_correlation,
        joint_stereo_detected,
        joint_stereo_confidence,
        hf_stereo_reduction,
        mid_side_ratio,
        stereo_narrowing_freq: narrowing_freq,
        evidence,
    }
}

/// Calculate stereo width from L/R channels
fn calculate_stereo_width(left: &[f32], right: &[f32]) -> f32 {
    // Stereo width based on difference signal energy
    let diff_energy: f32 = left.iter()
        .zip(right)
        .map(|(l, r)| (l - r).powi(2))
        .sum();
    
    let sum_energy: f32 = left.iter()
        .zip(right)
        .map(|(l, r)| (l + r).powi(2))
        .sum();
    
    if sum_energy < 1e-10 {
        return 0.0;
    }
    
    // Width = sqrt(diff_energy / sum_energy), clamped to [0, 1]
    (diff_energy / sum_energy).sqrt().min(1.0)
}

/// Calculate Pearson correlation coefficient between channels
fn calculate_correlation(left: &[f32], right: &[f32]) -> f32 {
    let n = left.len() as f32;
    
    let mean_l: f32 = left.iter().sum::<f32>() / n;
    let mean_r: f32 = right.iter().sum::<f32>() / n;
    
    let mut cov = 0.0f32;
    let mut var_l = 0.0f32;
    let mut var_r = 0.0f32;
    
    for (l, r) in left.iter().zip(right) {
        let dl = l - mean_l;
        let dr = r - mean_r;
        cov += dl * dr;
        var_l += dl * dl;
        var_r += dr * dr;
    }
    
    let denom = (var_l * var_r).sqrt();
    if denom < 1e-10 {
        return 1.0;  // Both channels constant
    }
    
    cov / denom
}

/// Analyze frequency-dependent stereo characteristics
fn analyze_frequency_dependent_stereo(
    left: &[f32],
    right: &[f32],
    _mid: &[f32],
    _side: &[f32],
    sample_rate: u32,
) -> (f32, Option<f32>) {
    let fft_size = 4096;
    let mut fft = FftProcessor::new(fft_size, WindowType::Hann);
    
    // Analyze multiple frames
    let num_frames = 20;
    let frame_hop = left.len().saturating_sub(fft_size) / num_frames.max(1);
    
    if frame_hop == 0 || left.len() < fft_size {
        return (0.0, None);
    }
    
    // Accumulate stereo width per frequency band
    let num_bands = 16;
    let mut band_widths: Vec<Vec<f32>> = vec![Vec::new(); num_bands];
    
    for frame_idx in 0..num_frames {
        let start = frame_idx * frame_hop;
        if start + fft_size > left.len() {
            break;
        }
        
        let left_frame = &left[start..start + fft_size];
        let right_frame = &right[start..start + fft_size];
        
        let left_spec = fft.magnitude_spectrum(left_frame);
        let right_spec = fft.magnitude_spectrum(right_frame);
        
        // Calculate stereo width per band
        let bins_per_band = left_spec.len() / num_bands;
        
        for band in 0..num_bands {
            let start_bin = band * bins_per_band;
            let end_bin = ((band + 1) * bins_per_band).min(left_spec.len());
            
            let mut diff_energy = 0.0f32;
            let mut sum_energy = 0.0f32;
            
            for bin in start_bin..end_bin {
                let l = left_spec[bin];
                let r = right_spec[bin];
                diff_energy += (l - r).powi(2);
                sum_energy += (l + r).powi(2);
            }
            
            if sum_energy > 1e-10 {
                let width = (diff_energy / sum_energy).sqrt().min(1.0);
                band_widths[band].push(width);
            }
        }
    }
    
    // Calculate average width per band
    let avg_band_widths: Vec<f32> = band_widths.iter()
        .map(|widths| {
            if widths.is_empty() {
                0.0
            } else {
                widths.iter().sum::<f32>() / widths.len() as f32
            }
        })
        .collect();
    
    // Compare low and high frequency stereo width
    let low_bands = &avg_band_widths[..num_bands / 4];
    let high_bands = &avg_band_widths[num_bands * 3 / 4..];
    
    let low_avg = if !low_bands.is_empty() {
        low_bands.iter().sum::<f32>() / low_bands.len() as f32
    } else {
        0.0
    };
    
    let high_avg = if !high_bands.is_empty() {
        high_bands.iter().sum::<f32>() / high_bands.len() as f32
    } else {
        0.0
    };
    
    // HF stereo reduction
    let hf_reduction = if low_avg > 0.01 {
        ((low_avg - high_avg) / low_avg).max(0.0)
    } else {
        0.0
    };
    
    // Find stereo narrowing frequency
    let narrowing_freq = find_stereo_narrowing_freq(&avg_band_widths, sample_rate, num_bands);
    
    (hf_reduction, narrowing_freq)
}

/// Find frequency where stereo starts to narrow significantly
fn find_stereo_narrowing_freq(
    band_widths: &[f32],
    sample_rate: u32,
    num_bands: usize,
) -> Option<f32> {
    if band_widths.is_empty() {
        return None;
    }
    
    // Find reference width from lower bands
    let reference_width = band_widths.iter()
        .take(band_widths.len() / 3)
        .cloned()
        .fold(0.0f32, f32::max);
    
    if reference_width < 0.05 {
        return None;  // Already narrow stereo
    }
    
    let threshold = reference_width * 0.5;  // 50% reduction threshold
    
    // Find first band where width drops below threshold
    for (i, &width) in band_widths.iter().enumerate() {
        if width < threshold && i > band_widths.len() / 4 {
            let freq = (i as f32 / num_bands as f32) * (sample_rate as f32 / 2.0);
            return Some(freq);
        }
    }
    
    None
}

/// Detect joint stereo encoding based on characteristics
fn detect_joint_stereo(
    stereo_width: f32,
    correlation: f32,
    hf_reduction: f32,
    ms_ratio: f32,
    narrowing_freq: Option<f32>,
) -> (bool, f32) {
    let mut score = 0.0f32;
    let mut max_score = 0.0f32;
    
    // High-frequency stereo reduction is strong indicator
    if hf_reduction > 0.3 {
        score += 30.0 * hf_reduction;
        max_score += 30.0;
    }
    max_score += 30.0;
    
    // Stereo narrowing at specific frequency
    if let Some(freq) = narrowing_freq {
        // Joint stereo typically narrows between 2kHz and 8kHz
        if freq > 2000.0 && freq < 10000.0 {
            score += 25.0;
        }
        max_score += 25.0;
    }
    
    // High M/S ratio suggests joint stereo processing
    if ms_ratio > 3.0 && ms_ratio < 20.0 {
        score += 15.0;
    }
    max_score += 15.0;
    
    // Very high correlation can indicate joint stereo or just correlated content
    if correlation > 0.9 {
        score += 10.0;
    }
    max_score += 10.0;
    
    // Moderate stereo width (not too wide, not mono)
    if stereo_width > 0.1 && stereo_width < 0.7 {
        score += 10.0;
    }
    max_score += 10.0;
    
    let confidence = score / max_score;
    let detected = confidence > 0.4 && hf_reduction > 0.2;
    
    (detected, confidence)
}

impl StereoAnalysis {
    fn mono_default() -> Self {
        StereoAnalysis {
            is_stereo: false,
            stereo_width: 0.0,
            channel_correlation: 1.0,
            joint_stereo_detected: false,
            joint_stereo_confidence: 0.0,
            hf_stereo_reduction: 0.0,
            mid_side_ratio: f32::INFINITY,
            stereo_narrowing_freq: None,
            evidence: vec!["Mono or invalid stereo".to_string()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_identical() {
        let samples = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let corr = calculate_correlation(&samples, &samples);
        assert!((corr - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_correlation_inverted() {
        let samples: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let inverted: Vec<f32> = samples.iter().map(|x| -x).collect();
        let corr = calculate_correlation(&samples, &inverted);
        assert!((corr - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_stereo_width_mono() {
        let samples = vec![0.5, 0.3, 0.1, -0.1, -0.3];
        let width = calculate_stereo_width(&samples, &samples);
        assert!(width < 0.001);
    }

    #[test]
    fn test_stereo_width_opposite() {
        let left = vec![0.5, 0.3, 0.1, -0.1, -0.3];
        let right: Vec<f32> = left.iter().map(|x| -x).collect();
        let width = calculate_stereo_width(&left, &right);
        assert!(width > 0.99);
    }
}
