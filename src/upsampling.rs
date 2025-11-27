// src/upsampling.rs
//
// Advanced upsampling detection using multiple methods:
// - Spectral analysis (frequency cutoff)
// - Null test (downsample-upsample comparison)
// - Inter-sample peak analysis

use crate::decoder::AudioData;
use crate::dsp::{
    FftProcessor, WindowType, upsample_sinc, downsample_simple,
    amplitude_to_db, peak_amplitude, rms,
};
use crate::spectral::SpectralAnalysis;

/// Upsampling detection result
#[derive(Debug, Clone)]
pub struct UpsamplingAnalysis {
    /// Whether upsampling was detected
    pub is_upsampled: bool,
    /// Detected original sample rate (if upsampled)
    pub original_sample_rate: Option<u32>,
    /// Current sample rate
    pub current_sample_rate: u32,
    /// Confidence in detection (0.0 to 1.0)
    pub confidence: f32,
    /// Method that detected upsampling
    pub detection_method: Option<String>,
    /// Detailed evidence
    pub evidence: Vec<String>,
    /// Results from individual detection methods
    pub method_results: UpsamplingMethodResults,
}

/// Results from individual detection methods
#[derive(Debug, Clone)]
pub struct UpsamplingMethodResults {
    /// Spectral method result
    pub spectral_detected: bool,
    pub spectral_original_rate: Option<u32>,
    pub spectral_confidence: f32,
    
    /// Null test method result  
    pub null_test_detected: bool,
    pub null_test_original_rate: Option<u32>,
    pub null_test_confidence: f32,
    
    /// Inter-sample peak method result
    pub isp_detected: bool,
    pub isp_confidence: f32,
}

/// Common sample rate pairs (original -> upsampled)
const SAMPLE_RATE_PAIRS: &[(u32, u32)] = &[
    (44100, 88200),
    (44100, 96000),
    (44100, 176400),
    (44100, 192000),
    (48000, 96000),
    (48000, 192000),
    (88200, 176400),
    (96000, 192000),
];

/// Analyze audio for upsampling
pub fn analyze_upsampling(audio: &AudioData, spectral: &SpectralAnalysis) -> UpsamplingAnalysis {
    let mut evidence = Vec::new();
    
    // Method 1: Spectral analysis
    let (spectral_detected, spectral_original, spectral_conf) = 
        detect_upsampling_spectral(audio, spectral);
    
    if spectral_detected {
        evidence.push(format!(
            "Spectral: Cutoff at {:.0} Hz suggests {} Hz source (confidence: {:.1}%)",
            spectral.frequency_cutoff,
            spectral_original.unwrap_or(0),
            spectral_conf * 100.0
        ));
    }
    
    // Method 2: Null test (for high sample rates only, to save computation)
    let (null_detected, null_original, null_conf) = if audio.sample_rate >= 88200 {
        detect_upsampling_null_test(audio)
    } else {
        (false, None, 0.0)
    };
    
    if null_detected {
        evidence.push(format!(
            "Null test: Audio matches {} Hz upsampled pattern (confidence: {:.1}%)",
            null_original.unwrap_or(0),
            null_conf * 100.0
        ));
    }
    
    // Method 3: Inter-sample peak analysis
    let (isp_detected, isp_conf) = detect_upsampling_intersample(audio);
    
    if isp_detected {
        evidence.push(format!(
            "Inter-sample peaks: Limited peak extension suggests upsampling (confidence: {:.1}%)",
            isp_conf * 100.0
        ));
    }
    
    // Combine results
    let method_results = UpsamplingMethodResults {
        spectral_detected,
        spectral_original_rate: spectral_original,
        spectral_confidence: spectral_conf,
        null_test_detected: null_detected,
        null_test_original_rate: null_original,
        null_test_confidence: null_conf,
        isp_detected,
        isp_confidence: isp_conf,
    };
    
    // Determine final result
    let (is_upsampled, original_rate, confidence, method) = 
        combine_detection_results(&method_results);
    
    if !is_upsampled {
        evidence.push("No upsampling detected".to_string());
    }
    
    UpsamplingAnalysis {
        is_upsampled,
        original_sample_rate: original_rate,
        current_sample_rate: audio.sample_rate,
        confidence,
        detection_method: method,
        evidence,
        method_results,
    }
}

/// Detect upsampling using spectral analysis
fn detect_upsampling_spectral(
    audio: &AudioData,
    spectral: &SpectralAnalysis,
) -> (bool, Option<u32>, f32) {
    let cutoff = spectral.frequency_cutoff;
    let nyquist = audio.sample_rate as f32 / 2.0;
    
    // Check if cutoff matches a lower sample rate's Nyquist
    for &(original, upsampled) in SAMPLE_RATE_PAIRS {
        if audio.sample_rate != upsampled {
            continue;
        }
        
        let original_nyquist = original as f32 / 2.0;
        let diff_ratio = (cutoff - original_nyquist).abs() / original_nyquist;
        
        // Allow 5% tolerance
        if diff_ratio < 0.05 {
            // Strong match
            let confidence = 0.9 * (1.0 - diff_ratio / 0.05);
            return (true, Some(original), confidence);
        } else if diff_ratio < 0.15 {
            // Weak match
            let confidence = 0.5 * (1.0 - diff_ratio / 0.15);
            return (true, Some(original), confidence);
        }
    }
    
    (false, None, 0.0)
}

/// Detect upsampling using null test (downsample and upsample back)
fn detect_upsampling_null_test(audio: &AudioData) -> (bool, Option<u32>, f32) {
    let mono = crate::decoder::extract_mono(audio);
    
    // Limit analysis to first 5 seconds for performance
    let max_samples = (audio.sample_rate * 5) as usize;
    let samples: Vec<f32> = mono.into_iter().take(max_samples).collect();
    
    let mut best_match: Option<(u32, f32)> = None;
    
    for &(original, upsampled) in SAMPLE_RATE_PAIRS {
        if audio.sample_rate != upsampled {
            continue;
        }
        
        // Calculate downsample factor
        let factor = upsampled / original;
        
        // Downsample
        let downsampled = downsample_simple(&samples, factor as usize);
        
        // Upsample back
        let upsampled_back = upsample_sinc(&downsampled, factor as usize);
        
        // Calculate correlation/similarity
        let similarity = calculate_spectral_similarity(&samples, &upsampled_back, audio.sample_rate);
        
        // High similarity = likely upsampled from this rate
        if similarity > 0.95 {
            let confidence = (similarity - 0.95) / 0.05;  // 0.95->0.0, 1.0->1.0
            
            if best_match.is_none() || confidence > best_match.unwrap().1 {
                best_match = Some((original, confidence));
            }
        }
    }
    
    match best_match {
        Some((rate, conf)) => (true, Some(rate), conf),
        None => (false, None, 0.0),
    }
}

/// Calculate spectral similarity between two signals
fn calculate_spectral_similarity(
    original: &[f32],
    processed: &[f32],
    sample_rate: u32,
) -> f32 {
    let fft_size = 4096;
    let mut fft = FftProcessor::new(fft_size, WindowType::Hann);
    
    let len = original.len().min(processed.len());
    if len < fft_size {
        return 0.0;
    }
    
    // Analyze multiple frames
    let num_frames = 10;
    let frame_hop = (len - fft_size) / num_frames;
    
    let mut similarities = Vec::new();
    
    for i in 0..num_frames {
        let start = i * frame_hop;
        if start + fft_size > len {
            break;
        }
        
        let orig_frame = &original[start..start + fft_size];
        let proc_frame = &processed[start..start + fft_size];
        
        let orig_spec = fft.magnitude_spectrum(orig_frame);
        let proc_spec = fft.magnitude_spectrum(proc_frame);
        
        // Correlation in log domain
        let orig_db: Vec<f32> = orig_spec.iter().map(|&m| amplitude_to_db(m.max(1e-10))).collect();
        let proc_db: Vec<f32> = proc_spec.iter().map(|&m| amplitude_to_db(m.max(1e-10))).collect();
        
        let similarity = spectral_correlation(&orig_db, &proc_db);
        similarities.push(similarity);
    }
    
    if similarities.is_empty() {
        return 0.0;
    }
    
    similarities.iter().sum::<f32>() / similarities.len() as f32
}

/// Calculate correlation between two spectra
fn spectral_correlation(spec1: &[f32], spec2: &[f32]) -> f32 {
    let n = spec1.len().min(spec2.len()) as f32;
    
    let mean1: f32 = spec1.iter().sum::<f32>() / n;
    let mean2: f32 = spec2.iter().sum::<f32>() / n;
    
    let mut cov = 0.0f32;
    let mut var1 = 0.0f32;
    let mut var2 = 0.0f32;
    
    for (s1, s2) in spec1.iter().zip(spec2) {
        let d1 = s1 - mean1;
        let d2 = s2 - mean2;
        cov += d1 * d2;
        var1 += d1 * d1;
        var2 += d2 * d2;
    }
    
    let denom = (var1 * var2).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }
    
    (cov / denom).max(0.0)
}

/// Detect upsampling using inter-sample peak analysis
/// True high-sample-rate content has energy between sample points
fn detect_upsampling_intersample(audio: &AudioData) -> (bool, f32) {
    // Only relevant for high sample rates
    if audio.sample_rate < 88200 {
        return (false, 0.0);
    }
    
    let mono = crate::decoder::extract_mono(audio);
    
    // Analyze segments with transients (where ISP matters most)
    let segment_size = 4096;
    let num_segments = (mono.len() / segment_size).min(50);
    
    let mut isp_ratios = Vec::new();
    
    for i in 0..num_segments {
        let start = i * mono.len() / num_segments;
        let end = (start + segment_size).min(mono.len());
        let segment = &mono[start..end];
        
        // Skip quiet segments
        let segment_peak = peak_amplitude(segment);
        if segment_peak < 0.1 {
            continue;
        }
        
        // Calculate sample peak
        let sample_peak = segment_peak;
        
        // Upsample by 4x and measure true peak
        let upsampled = upsample_sinc(segment, 4);
        let true_peak = peak_amplitude(&upsampled);
        
        // Ratio of true peak to sample peak
        // True high-res audio: ratio can be > 1.0 (inter-sample overs)
        // Upsampled audio: ratio ≈ 1.0 (no new information between samples)
        let ratio = true_peak / sample_peak;
        isp_ratios.push(ratio);
    }
    
    if isp_ratios.is_empty() {
        return (false, 0.0);
    }
    
    // Average ISP ratio
    let avg_ratio: f32 = isp_ratios.iter().sum::<f32>() / isp_ratios.len() as f32;
    
    // Maximum ISP ratio (should be > 1.0 for true high-res)
    let max_ratio = isp_ratios.iter().cloned().fold(0.0f32, f32::max);
    
    // If max ratio never exceeds ~1.02, audio is likely upsampled
    // True high-res typically has max ratio > 1.1
    if max_ratio < 1.02 && avg_ratio < 1.01 {
        let confidence = 1.0 - (max_ratio - 1.0) * 50.0;  // Higher confidence when ratio closer to 1.0
        (true, confidence.max(0.0).min(0.8))  // Cap at 0.8, this method alone isn't definitive
    } else {
        (false, 0.0)
    }
}

/// Combine results from multiple detection methods
fn combine_detection_results(
    results: &UpsamplingMethodResults,
) -> (bool, Option<u32>, f32, Option<String>) {
    let mut detections = Vec::new();
    
    if results.spectral_detected {
        detections.push((
            results.spectral_original_rate,
            results.spectral_confidence,
            "Spectral analysis",
        ));
    }
    
    if results.null_test_detected {
        detections.push((
            results.null_test_original_rate,
            results.null_test_confidence,
            "Null test",
        ));
    }
    
    if results.isp_detected {
        detections.push((
            None,  // ISP doesn't determine specific rate
            results.isp_confidence,
            "Inter-sample peaks",
        ));
    }
    
    if detections.is_empty() {
        return (false, None, 0.0, None);
    }
    
    // Sort by confidence
    detections.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    let best = &detections[0];
    
    // Combine confidences if multiple methods agree
    let mut combined_confidence = best.1;
    for detection in &detections[1..] {
        if detection.0 == best.0 || detection.0.is_none() {
            combined_confidence = 1.0 - (1.0 - combined_confidence) * (1.0 - detection.1 * 0.5);
        }
    }
    
    (true, best.0, combined_confidence.min(0.95), Some(best.2.to_string()))
}

/// Detect specific upsampling ratio
pub fn detect_upsampling_ratio(audio: &AudioData) -> Option<(u32, u32)> {
    for &(original, upsampled) in SAMPLE_RATE_PAIRS {
        if audio.sample_rate == upsampled {
            // Quick check: does Nyquist of original match content cutoff?
            let mono = crate::decoder::extract_mono(audio);
            let fft_size = 8192;
            
            if mono.len() < fft_size {
                continue;
            }
            
            let mut fft = FftProcessor::new(fft_size, WindowType::BlackmanHarris);
            
            // Analyze middle of track
            let start = mono.len() / 2;
            let spectrum = fft.magnitude_spectrum(&mono[start..start + fft_size]);
            
            // Find -60dB point from peak
            let peak = spectrum.iter().cloned().fold(0.0f32, f32::max);
            let threshold = peak * 0.001;  // -60dB
            
            let nyquist = audio.sample_rate as f32 / 2.0;
            let bin_hz = nyquist / spectrum.len() as f32;
            
            // Search from high frequencies down
            for (i, &mag) in spectrum.iter().enumerate().rev() {
                if mag > threshold {
                    let cutoff_hz = i as f32 * bin_hz;
                    let original_nyquist = original as f32 / 2.0;
                    
                    // Within 10% of original Nyquist?
                    if (cutoff_hz - original_nyquist).abs() / original_nyquist < 0.1 {
                        return Some((original, upsampled));
                    }
                    break;
                }
            }
        }
    }
    
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_correlation_identical() {
        let spec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = spectral_correlation(&spec, &spec);
        assert!((corr - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_spectral_correlation_different() {
        let spec1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let spec2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr = spectral_correlation(&spec1, &spec2);
        assert!((corr - (-1.0)).abs() < 0.001);  // Inverse correlation
    }
}
