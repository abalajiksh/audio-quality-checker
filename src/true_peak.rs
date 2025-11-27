// src/true_peak.rs
//
// True Peak analysis following ITU-R BS.1770 principles.
// Detects inter-sample peaks and clipping artifacts.

use crate::decoder::AudioData;
use crate::dsp::{upsample_sinc, amplitude_to_db, peak_amplitude};

/// True peak analysis result
#[derive(Debug, Clone)]
pub struct TruePeakAnalysis {
    /// Sample peak in dBFS
    pub sample_peak_dbfs: f32,
    /// True peak in dBFS (after oversampling)
    pub true_peak_dbfs: f32,
    /// Inter-sample peak margin (true_peak - sample_peak)
    pub inter_sample_margin: f32,
    /// Whether inter-sample overs are present (true_peak > 0 dBFS)
    pub has_inter_sample_overs: bool,
    /// Number of inter-sample overs detected
    pub inter_sample_over_count: usize,
    /// Maximum over level
    pub max_over_level: f32,
    /// Whether clipping is detected
    pub has_clipping: bool,
    /// Percentage of samples at full scale
    pub clipping_percentage: f32,
    /// Loudness analysis
    pub loudness_info: LoudnessInfo,
    /// Evidence strings
    pub evidence: Vec<String>,
}

/// Basic loudness information
#[derive(Debug, Clone)]
pub struct LoudnessInfo {
    /// RMS level in dBFS
    pub rms_dbfs: f32,
    /// Crest factor (peak/RMS ratio in dB)
    pub crest_factor_db: f32,
    /// Dynamic range estimate
    pub dynamic_range_db: f32,
}

/// Perform true peak analysis
pub fn analyze_true_peak(audio: &AudioData) -> TruePeakAnalysis {
    let mut evidence = Vec::new();
    
    // Extract mono for analysis (or analyze both channels and take max)
    let mono = crate::decoder::extract_mono(audio);
    
    // Sample peak
    let sample_peak = peak_amplitude(&mono);
    let sample_peak_dbfs = amplitude_to_db(sample_peak);
    
    evidence.push(format!("Sample peak: {:.2} dBFS", sample_peak_dbfs));
    
    // True peak via 4x oversampling (ITU-R BS.1770)
    let (true_peak, inter_sample_overs, max_over) = calculate_true_peak(&mono, 4);
    let true_peak_dbfs = amplitude_to_db(true_peak);
    
    evidence.push(format!("True peak: {:.2} dBFS", true_peak_dbfs));
    
    let inter_sample_margin = true_peak_dbfs - sample_peak_dbfs;
    evidence.push(format!("Inter-sample margin: {:.2} dB", inter_sample_margin));
    
    let has_inter_sample_overs = true_peak_dbfs > 0.0;
    if has_inter_sample_overs {
        evidence.push(format!("WARNING: {} inter-sample overs detected", inter_sample_overs));
    }
    
    // Check for clipping
    let (has_clipping, clipping_pct) = detect_clipping(&mono);
    if has_clipping {
        evidence.push(format!("Clipping detected: {:.3}% of samples at full scale", clipping_pct * 100.0));
    }
    
    // Loudness analysis
    let loudness_info = analyze_loudness(&mono);
    evidence.push(format!("RMS level: {:.2} dBFS", loudness_info.rms_dbfs));
    evidence.push(format!("Crest factor: {:.2} dB", loudness_info.crest_factor_db));
    evidence.push(format!("Dynamic range: {:.2} dB", loudness_info.dynamic_range_db));
    
    TruePeakAnalysis {
        sample_peak_dbfs,
        true_peak_dbfs,
        inter_sample_margin,
        has_inter_sample_overs,
        inter_sample_over_count: inter_sample_overs,
        max_over_level: max_over,
        has_clipping,
        clipping_percentage: clipping_pct,
        loudness_info,
        evidence,
    }
}

/// Calculate true peak using oversampling
fn calculate_true_peak(samples: &[f32], oversample_factor: usize) -> (f32, usize, f32) {
    // Process in chunks to manage memory
    let chunk_size = 4096;
    let mut max_true_peak = 0.0f32;
    let mut over_count = 0usize;
    let mut max_over = 0.0f32;
    
    for chunk in samples.chunks(chunk_size) {
        let oversampled = upsample_sinc(chunk, oversample_factor);
        
        for &sample in &oversampled {
            let abs_sample = sample.abs();
            max_true_peak = max_true_peak.max(abs_sample);
            
            if abs_sample > 1.0 {
                over_count += 1;
                max_over = max_over.max(abs_sample - 1.0);
            }
        }
    }
    
    // Adjust over count for oversampling
    let adjusted_over_count = over_count / oversample_factor;
    
    (max_true_peak, adjusted_over_count, amplitude_to_db(1.0 + max_over))
}

/// Detect clipping (samples at or near full scale)
fn detect_clipping(samples: &[f32]) -> (bool, f32) {
    let clipping_threshold = 0.999;  // Very close to full scale
    
    let clipped_count = samples.iter()
        .filter(|&&s| s.abs() >= clipping_threshold)
        .count();
    
    let clipping_percentage = clipped_count as f32 / samples.len() as f32;
    
    // Consider it clipping if more than 0.01% of samples are at full scale
    let has_clipping = clipping_percentage > 0.0001;
    
    (has_clipping, clipping_percentage)
}

/// Analyze loudness characteristics
fn analyze_loudness(samples: &[f32]) -> LoudnessInfo {
    // RMS calculation
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    let rms = (sum_sq / samples.len() as f32).sqrt();
    let rms_dbfs = amplitude_to_db(rms);
    
    // Peak
    let peak = peak_amplitude(samples);
    let peak_dbfs = amplitude_to_db(peak);
    
    // Crest factor
    let crest_factor_db = peak_dbfs - rms_dbfs;
    
    // Dynamic range estimation using percentiles
    let dynamic_range_db = estimate_dynamic_range(samples);
    
    LoudnessInfo {
        rms_dbfs,
        crest_factor_db,
        dynamic_range_db,
    }
}

/// Estimate dynamic range using percentile method
fn estimate_dynamic_range(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    
    // Calculate RMS in short windows
    let window_size = 1024;
    let hop = 512;
    
    let mut rms_values: Vec<f32> = Vec::new();
    
    for chunk_start in (0..samples.len().saturating_sub(window_size)).step_by(hop) {
        let chunk = &samples[chunk_start..chunk_start + window_size];
        let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / window_size as f32).sqrt();
        if rms > 1e-8 {  // Skip digital silence
            rms_values.push(amplitude_to_db(rms));
        }
    }
    
    if rms_values.len() < 10 {
        return 0.0;
    }
    
    rms_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    // 95th percentile - 5th percentile
    let p95_idx = (rms_values.len() * 95) / 100;
    let p5_idx = (rms_values.len() * 5) / 100;
    
    rms_values[p95_idx] - rms_values[p5_idx]
}

/// Per-channel true peak analysis for stereo files
#[derive(Debug, Clone)]
pub struct ChannelTruePeak {
    pub left_sample_peak_dbfs: f32,
    pub left_true_peak_dbfs: f32,
    pub right_sample_peak_dbfs: f32,
    pub right_true_peak_dbfs: f32,
    pub stereo_balance: f32,  // Negative = left louder, positive = right louder
}

/// Analyze true peak per channel
pub fn analyze_true_peak_stereo(audio: &AudioData) -> Option<ChannelTruePeak> {
    if audio.channels < 2 {
        return None;
    }
    
    let (left, right) = crate::decoder::extract_stereo(audio)?;
    
    let left_sample_peak = peak_amplitude(&left);
    let right_sample_peak = peak_amplitude(&right);
    
    let (left_true_peak, _, _) = calculate_true_peak(&left, 4);
    let (right_true_peak, _, _) = calculate_true_peak(&right, 4);
    
    let left_sample_peak_dbfs = amplitude_to_db(left_sample_peak);
    let right_sample_peak_dbfs = amplitude_to_db(right_sample_peak);
    let left_true_peak_dbfs = amplitude_to_db(left_true_peak);
    let right_true_peak_dbfs = amplitude_to_db(right_true_peak);
    
    // Stereo balance based on RMS
    let left_rms = crate::dsp::rms(&left);
    let right_rms = crate::dsp::rms(&right);
    
    let stereo_balance = if left_rms > 0.0 && right_rms > 0.0 {
        amplitude_to_db(right_rms) - amplitude_to_db(left_rms)
    } else {
        0.0
    };
    
    Some(ChannelTruePeak {
        left_sample_peak_dbfs,
        left_true_peak_dbfs,
        right_sample_peak_dbfs,
        right_true_peak_dbfs,
        stereo_balance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clipping_detection_no_clipping() {
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let (has_clipping, _) = detect_clipping(&samples);
        assert!(!has_clipping);
    }

    #[test]
    fn test_clipping_detection_with_clipping() {
        let mut samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        // Add some clipped samples
        for i in 0..10 {
            samples[i] = 1.0;
        }
        let (has_clipping, _) = detect_clipping(&samples);
        assert!(has_clipping);
    }
}
