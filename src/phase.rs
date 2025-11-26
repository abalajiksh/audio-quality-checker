// src/phase.rs
//
// Phase analysis for detecting codec artifacts.
// Lossy codecs can introduce phase discontinuities at frame boundaries.

use crate::decoder::AudioData;
use crate::dsp::{FftProcessor, WindowType};
use rustfft::num_complex::Complex;
use std::f32::consts::PI;

/// Phase analysis result
#[derive(Debug, Clone)]
pub struct PhaseAnalysis {
    /// Average phase coherence (0.0 to 1.0)
    pub phase_coherence: f32,
    /// Phase discontinuity score (higher = more discontinuities)
    pub discontinuity_score: f32,
    /// Number of detected phase jumps
    pub phase_jump_count: usize,
    /// Whether codec phase artifacts are likely
    pub codec_artifacts_likely: bool,
    /// Confidence in analysis
    pub confidence: f32,
    /// Evidence strings
    pub evidence: Vec<String>,
}

/// Phase information for a single frame
#[derive(Debug, Clone)]
struct FramePhase {
    /// Phase values per frequency bin
    phases: Vec<f32>,
    /// Magnitude values for weighting
    magnitudes: Vec<f32>,
}

/// Analyze phase characteristics of audio
pub fn analyze_phase(audio: &AudioData) -> PhaseAnalysis {
    let mono = crate::decoder::extract_mono(audio);
    
    let fft_size = 2048;
    let hop_size = fft_size / 2;  // 50% overlap for phase analysis
    
    if mono.len() < fft_size * 4 {
        return PhaseAnalysis {
            phase_coherence: 0.0,
            discontinuity_score: 0.0,
            phase_jump_count: 0,
            codec_artifacts_likely: false,
            confidence: 0.0,
            evidence: vec!["Audio too short for phase analysis".to_string()],
        };
    }
    
    let mut evidence = Vec::new();
    
    // Compute phase for multiple frames
    let frame_phases = compute_frame_phases(&mono, fft_size, hop_size, audio.sample_rate);
    
    if frame_phases.len() < 10 {
        return PhaseAnalysis {
            phase_coherence: 0.0,
            discontinuity_score: 0.0,
            phase_jump_count: 0,
            codec_artifacts_likely: false,
            confidence: 0.0,
            evidence: vec!["Not enough frames for analysis".to_string()],
        };
    }
    
    evidence.push(format!("Analyzed {} frames", frame_phases.len()));
    
    // Analyze phase coherence between consecutive frames
    let (coherence, discontinuities) = analyze_phase_coherence(&frame_phases, hop_size, audio.sample_rate);
    
    // Count significant phase jumps
    let phase_jumps = count_phase_jumps(&frame_phases);
    
    evidence.push(format!("Phase coherence: {:.3}", coherence));
    evidence.push(format!("Discontinuity score: {:.3}", discontinuities));
    evidence.push(format!("Phase jumps detected: {}", phase_jumps));
    
    // Determine if codec artifacts are present
    let codec_artifacts_likely = discontinuities > 0.3 && phase_jumps > frame_phases.len() / 10;
    
    if codec_artifacts_likely {
        evidence.push("Pattern suggests lossy codec processing".to_string());
    }
    
    // Confidence based on number of frames
    let confidence = if frame_phases.len() >= 100 {
        0.9
    } else if frame_phases.len() >= 50 {
        0.75
    } else {
        0.5
    };
    
    PhaseAnalysis {
        phase_coherence: coherence,
        discontinuity_score: discontinuities,
        phase_jump_count: phase_jumps,
        codec_artifacts_likely,
        confidence,
        evidence,
    }
}

/// Compute phase information for all frames
fn compute_frame_phases(
    samples: &[f32],
    fft_size: usize,
    hop_size: usize,
    _sample_rate: u32,
) -> Vec<FramePhase> {
    let mut fft = FftProcessor::new(fft_size, WindowType::Hann);
    let num_frames = (samples.len() - fft_size) / hop_size;
    
    let mut frames = Vec::with_capacity(num_frames);
    
    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        if start + fft_size > samples.len() {
            break;
        }
        
        let frame = &samples[start..start + fft_size];
        let complex_spectrum = fft.complex_spectrum(frame);
        
        let phases: Vec<f32> = complex_spectrum.iter()
            .map(|c| c.arg())  // Extract phase angle
            .collect();
        
        let magnitudes: Vec<f32> = complex_spectrum.iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();
        
        frames.push(FramePhase { phases, magnitudes });
    }
    
    frames
}

/// Analyze phase coherence between consecutive frames
fn analyze_phase_coherence(
    frames: &[FramePhase],
    hop_size: usize,
    sample_rate: u32,
) -> (f32, f32) {
    if frames.len() < 2 {
        return (1.0, 0.0);
    }
    
    let mut coherence_sum = 0.0f32;
    let mut discontinuity_sum = 0.0f32;
    let mut total_weight = 0.0f32;
    
    // Expected phase advance per frame for each bin
    // phase_advance = 2π * frequency * hop_time
    let hop_time = hop_size as f32 / sample_rate as f32;
    
    for i in 1..frames.len() {
        let prev = &frames[i - 1];
        let curr = &frames[i];
        
        if prev.phases.len() != curr.phases.len() {
            continue;
        }
        
        for bin in 1..prev.phases.len() {  // Skip DC bin
            // Frequency of this bin
            let freq = bin as f32 * sample_rate as f32 / (2.0 * prev.phases.len() as f32);
            
            // Expected phase advance
            let expected_advance = 2.0 * PI * freq * hop_time;
            
            // Actual phase advance
            let actual_advance = wrap_phase(curr.phases[bin] - prev.phases[bin]);
            
            // Phase prediction error
            let error = wrap_phase(actual_advance - expected_advance).abs();
            
            // Weight by magnitude (only consider audible components)
            let weight = (prev.magnitudes[bin] + curr.magnitudes[bin]) / 2.0;
            
            if weight > 1e-6 {
                // Coherence: how well does phase follow expectation
                let bin_coherence = (PI - error) / PI;  // 1.0 = perfect, 0.0 = worst
                coherence_sum += bin_coherence * weight;
                
                // Discontinuity: large phase errors
                if error > PI / 2.0 {
                    discontinuity_sum += weight;
                }
                
                total_weight += weight;
            }
        }
    }
    
    if total_weight < 1e-10 {
        return (1.0, 0.0);
    }
    
    let coherence = coherence_sum / total_weight;
    let discontinuity = discontinuity_sum / total_weight;
    
    (coherence, discontinuity)
}

/// Wrap phase to [-π, π]
fn wrap_phase(phase: f32) -> f32 {
    let mut p = phase;
    while p > PI {
        p -= 2.0 * PI;
    }
    while p < -PI {
        p += 2.0 * PI;
    }
    p
}

/// Count significant phase jumps
fn count_phase_jumps(frames: &[FramePhase]) -> usize {
    if frames.len() < 2 {
        return 0;
    }
    
    let mut jump_count = 0;
    let jump_threshold = PI * 0.75;  // 135 degrees
    
    for i in 1..frames.len() {
        let prev = &frames[i - 1];
        let curr = &frames[i];
        
        // Check middle frequency bins (most sensitive)
        let start_bin = prev.phases.len() / 4;
        let end_bin = prev.phases.len() * 3 / 4;
        
        let mut bin_jumps = 0;
        let mut total_bins = 0;
        
        for bin in start_bin..end_bin.min(curr.phases.len()) {
            let weight = (prev.magnitudes[bin] + curr.magnitudes[bin]) / 2.0;
            
            if weight > 0.001 {  // Only consider audible bins
                let phase_diff = wrap_phase(curr.phases[bin] - prev.phases[bin]).abs();
                if phase_diff > jump_threshold {
                    bin_jumps += 1;
                }
                total_bins += 1;
            }
        }
        
        // A frame transition counts as a jump if many bins have large phase changes
        if total_bins > 0 && bin_jumps as f32 / total_bins as f32 > 0.3 {
            jump_count += 1;
        }
    }
    
    jump_count
}

/// Instantaneous frequency analysis
/// Can reveal codec processing artifacts
#[derive(Debug, Clone)]
pub struct InstantaneousFrequencyAnalysis {
    /// Frequency deviation from expected (weighted average)
    pub frequency_deviation: f32,
    /// Variance in instantaneous frequency
    pub frequency_variance: f32,
    /// Score indicating processing artifacts
    pub artifact_score: f32,
}

/// Analyze instantaneous frequency characteristics
pub fn analyze_instantaneous_frequency(audio: &AudioData) -> InstantaneousFrequencyAnalysis {
    let mono = crate::decoder::extract_mono(audio);
    
    let fft_size = 2048;
    let hop_size = fft_size / 4;
    
    let frames = compute_frame_phases(&mono, fft_size, hop_size, audio.sample_rate);
    
    if frames.len() < 10 {
        return InstantaneousFrequencyAnalysis {
            frequency_deviation: 0.0,
            frequency_variance: 0.0,
            artifact_score: 0.0,
        };
    }
    
    let hop_time = hop_size as f32 / audio.sample_rate as f32;
    let mut deviations = Vec::new();
    
    for i in 1..frames.len() {
        let prev = &frames[i - 1];
        let curr = &frames[i];
        
        for bin in 1..prev.phases.len().min(curr.phases.len()) {
            let weight = (prev.magnitudes[bin] + curr.magnitudes[bin]) / 2.0;
            
            if weight > 0.01 {
                // Expected frequency for this bin
                let expected_freq = bin as f32 * audio.sample_rate as f32 
                    / (2.0 * prev.phases.len() as f32);
                
                // Instantaneous frequency from phase derivative
                let phase_diff = wrap_phase(curr.phases[bin] - prev.phases[bin]);
                let inst_freq = phase_diff / (2.0 * PI * hop_time);
                
                // Deviation (should be close to 0 for clean audio)
                let deviation = (inst_freq - expected_freq) / expected_freq;
                deviations.push(deviation.abs());
            }
        }
    }
    
    if deviations.is_empty() {
        return InstantaneousFrequencyAnalysis {
            frequency_deviation: 0.0,
            frequency_variance: 0.0,
            artifact_score: 0.0,
        };
    }
    
    let mean_deviation = deviations.iter().sum::<f32>() / deviations.len() as f32;
    let variance = deviations.iter()
        .map(|d| (d - mean_deviation).powi(2))
        .sum::<f32>() / deviations.len() as f32;
    
    // High variance suggests codec processing
    let artifact_score = (variance * 100.0).min(1.0);
    
    InstantaneousFrequencyAnalysis {
        frequency_deviation: mean_deviation,
        frequency_variance: variance,
        artifact_score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_phase() {
        assert!((wrap_phase(0.0)).abs() < 0.001);
        assert!((wrap_phase(PI) - PI).abs() < 0.001);
        assert!((wrap_phase(-PI) - (-PI)).abs() < 0.001);
        assert!((wrap_phase(3.0 * PI) - PI).abs() < 0.001);
        assert!((wrap_phase(-3.0 * PI) - (-PI)).abs() < 0.001);
    }
}
