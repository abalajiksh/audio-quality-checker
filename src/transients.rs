// src/transients.rs
//
// Transient analysis and pre-echo detection.
// Pre-echo is a characteristic artifact of transform-based lossy codecs (MP3, AAC, etc.)

use crate::decoder::AudioData;
use crate::dsp::{find_transients, amplitude_to_db, rms};

/// Pre-echo analysis result
#[derive(Debug, Clone)]
pub struct PreEchoAnalysis {
    /// Number of transients detected
    pub transient_count: usize,
    /// Number of transients with pre-echo
    pub pre_echo_count: usize,
    /// Average pre-echo level in dB (relative to transient)
    pub avg_pre_echo_level: f32,
    /// Maximum pre-echo level detected
    pub max_pre_echo_level: f32,
    /// Pre-echo score (0.0 = no pre-echo, 1.0 = severe pre-echo)
    pub pre_echo_score: f32,
    /// Whether lossy codec pre-echo pattern is detected
    pub lossy_pre_echo_detected: bool,
    /// Confidence in detection
    pub confidence: f32,
    /// Evidence strings
    pub evidence: Vec<String>,
}

/// Transient characteristics
#[derive(Debug, Clone)]
pub struct TransientInfo {
    /// Sample position of transient
    pub position: usize,
    /// Peak level at transient
    pub peak_level: f32,
    /// Pre-transient energy level
    pub pre_energy: f32,
    /// Pre-echo detected for this transient
    pub has_pre_echo: bool,
    /// Pre-echo level in dB relative to transient
    pub pre_echo_level_db: f32,
}

/// Analyze pre-echo artifacts
pub fn analyze_pre_echo(audio: &AudioData) -> PreEchoAnalysis {
    let mono = crate::decoder::extract_mono(audio);
    
    // Find transients
    let transient_positions = find_transients(&mono, 15.0, audio.sample_rate as usize / 20);
    
    if transient_positions.is_empty() {
        return PreEchoAnalysis {
            transient_count: 0,
            pre_echo_count: 0,
            avg_pre_echo_level: -120.0,
            max_pre_echo_level: -120.0,
            pre_echo_score: 0.0,
            lossy_pre_echo_detected: false,
            confidence: 0.0,
            evidence: vec!["No transients detected for analysis".to_string()],
        };
    }
    
    let mut evidence = Vec::new();
    evidence.push(format!("Detected {} transients", transient_positions.len()));
    
    // Analyze each transient for pre-echo
    let transient_infos: Vec<TransientInfo> = transient_positions.iter()
        .filter_map(|&pos| analyze_single_transient(&mono, pos, audio.sample_rate))
        .collect();
    
    let transient_count = transient_infos.len();
    
    if transient_count == 0 {
        return PreEchoAnalysis {
            transient_count: 0,
            pre_echo_count: 0,
            avg_pre_echo_level: -120.0,
            max_pre_echo_level: -120.0,
            pre_echo_score: 0.0,
            lossy_pre_echo_detected: false,
            confidence: 0.0,
            evidence: vec!["No analyzable transients".to_string()],
        };
    }
    
    // Count transients with pre-echo
    let pre_echo_transients: Vec<&TransientInfo> = transient_infos.iter()
        .filter(|t| t.has_pre_echo)
        .collect();
    
    let pre_echo_count = pre_echo_transients.len();
    
    // Calculate statistics
    let pre_echo_levels: Vec<f32> = pre_echo_transients.iter()
        .map(|t| t.pre_echo_level_db)
        .collect();
    
    let avg_pre_echo_level = if pre_echo_levels.is_empty() {
        -120.0
    } else {
        pre_echo_levels.iter().sum::<f32>() / pre_echo_levels.len() as f32
    };
    
    let max_pre_echo_level = pre_echo_levels.iter()
        .cloned()
        .fold(f32::MIN, f32::max);
    
    // Calculate pre-echo score
    let pre_echo_ratio = pre_echo_count as f32 / transient_count as f32;
    let level_factor = if avg_pre_echo_level > -50.0 {
        (avg_pre_echo_level + 50.0) / 30.0  // -20dB pre-echo = 1.0
    } else {
        0.0
    };
    
    let pre_echo_score = (pre_echo_ratio * 0.6 + level_factor * 0.4).min(1.0);
    
    // Determine if this matches lossy codec pattern
    let lossy_pre_echo_detected = pre_echo_ratio > 0.3 
        && avg_pre_echo_level > -40.0 
        && transient_count >= 5;
    
    // Confidence based on sample size
    let confidence = if transient_count >= 20 {
        0.9
    } else if transient_count >= 10 {
        0.75
    } else if transient_count >= 5 {
        0.6
    } else {
        0.4
    };
    
    evidence.push(format!("{} of {} transients show pre-echo", pre_echo_count, transient_count));
    evidence.push(format!("Average pre-echo level: {:.1} dB", avg_pre_echo_level));
    
    if lossy_pre_echo_detected {
        evidence.push("Pattern consistent with lossy codec artifacts".to_string());
    }
    
    PreEchoAnalysis {
        transient_count,
        pre_echo_count,
        avg_pre_echo_level,
        max_pre_echo_level,
        pre_echo_score,
        lossy_pre_echo_detected,
        confidence,
        evidence,
    }
}

/// Analyze a single transient for pre-echo
fn analyze_single_transient(
    samples: &[f32],
    position: usize,
    sample_rate: u32,
) -> Option<TransientInfo> {
    // Pre-echo window: typically 20-30ms before transient for MP3
    // This corresponds to the MDCT window size
    let pre_echo_window_ms = 25.0;
    let pre_echo_samples = (sample_rate as f32 * pre_echo_window_ms / 1000.0) as usize;
    
    // Analysis windows
    let analysis_window = pre_echo_samples / 4;  // Shorter analysis chunks
    
    // Need enough samples before the transient
    if position < pre_echo_samples + analysis_window {
        return None;
    }
    
    // Get the transient level
    let transient_start = position;
    let transient_end = (position + analysis_window).min(samples.len());
    let transient_window = &samples[transient_start..transient_end];
    let transient_level = transient_window.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    
    if transient_level < 0.01 {
        return None;  // Too quiet to analyze
    }
    
    let transient_db = amplitude_to_db(transient_level);
    
    // Analyze the pre-transient region
    // We look at multiple windows before the transient
    let num_pre_windows = 4;
    let mut pre_levels = Vec::new();
    
    for i in 0..num_pre_windows {
        let window_end = position - i * analysis_window;
        let window_start = window_end.saturating_sub(analysis_window);
        
        if window_start >= samples.len() || window_end > samples.len() {
            continue;
        }
        
        let window = &samples[window_start..window_end];
        let level = rms(window);
        pre_levels.push(level);
    }
    
    if pre_levels.len() < 2 {
        return None;
    }
    
    // The window immediately before the transient
    let immediate_pre_level = pre_levels[0];
    
    // Earlier windows (should be quieter if no pre-echo)
    let earlier_pre_level: f32 = pre_levels[1..].iter().sum::<f32>() / (pre_levels.len() - 1) as f32;
    
    // Pre-echo detection:
    // - True silence before transient: immediate_pre_level ≈ earlier_pre_level (both low)
    // - Pre-echo present: immediate_pre_level > earlier_pre_level * threshold
    
    let pre_echo_ratio = if earlier_pre_level > 1e-8 {
        immediate_pre_level / earlier_pre_level
    } else {
        1.0
    };
    
    // Also check absolute level - pre-echo should be audible
    let immediate_pre_db = amplitude_to_db(immediate_pre_level);
    let pre_echo_relative_db = immediate_pre_db - transient_db;
    
    // Pre-echo typically 20-40dB below transient, but audible
    let has_pre_echo = pre_echo_ratio > 3.0  // 3x more energy than earlier
        && pre_echo_relative_db > -50.0  // Within 50dB of transient
        && pre_echo_relative_db < -10.0  // But not too close (that would be attack)
        && immediate_pre_level > 0.001;  // Absolute threshold
    
    Some(TransientInfo {
        position,
        peak_level: transient_level,
        pre_energy: immediate_pre_level,
        has_pre_echo,
        pre_echo_level_db: pre_echo_relative_db,
    })
}

/// Frame boundary analysis for codec detection
/// Lossy codecs process audio in fixed-size frames, which can leave artifacts
#[derive(Debug, Clone)]
pub struct FrameBoundaryAnalysis {
    /// Detected frame size in samples (if any)
    pub detected_frame_size: Option<usize>,
    /// Confidence in frame size detection
    pub frame_detection_confidence: f32,
    /// Periodic artifact score
    pub periodic_artifact_score: f32,
    /// Evidence strings
    pub evidence: Vec<String>,
}

/// Analyze for codec frame boundaries
pub fn analyze_frame_boundaries(audio: &AudioData) -> FrameBoundaryAnalysis {
    let mono = crate::decoder::extract_mono(audio);
    
    // Common codec frame sizes (in samples at various sample rates)
    let common_frame_sizes: Vec<usize> = vec![
        576,   // MP3 (half frame for short blocks)
        1152,  // MP3 standard frame
        1024,  // AAC
        2048,  // AAC long block
        960,   // Opus 20ms at 48kHz
        480,   // Opus 10ms at 48kHz
        2880,  // Opus 60ms at 48kHz
    ];
    
    let mut evidence = Vec::new();
    let mut best_frame_size: Option<usize> = None;
    let mut best_score = 0.0f32;
    
    for &frame_size in &common_frame_sizes {
        let score = detect_frame_periodicity(&mono, frame_size);
        
        if score > best_score && score > 0.3 {
            best_score = score;
            best_frame_size = Some(frame_size);
        }
    }
    
    if let Some(frame_size) = best_frame_size {
        evidence.push(format!("Possible frame size: {} samples", frame_size));
        evidence.push(format!("Periodicity score: {:.2}", best_score));
        
        // Identify likely codec
        let codec_guess = match frame_size {
            576 | 1152 => "MP3",
            1024 | 2048 => "AAC",
            480 | 960 | 2880 => "Opus",
            _ => "Unknown",
        };
        evidence.push(format!("Likely codec: {}", codec_guess));
    }
    
    FrameBoundaryAnalysis {
        detected_frame_size: best_frame_size,
        frame_detection_confidence: best_score,
        periodic_artifact_score: best_score,
        evidence,
    }
}

/// Detect periodicity at a specific frame size
fn detect_frame_periodicity(samples: &[f32], frame_size: usize) -> f32 {
    if samples.len() < frame_size * 10 {
        return 0.0;
    }
    
    // Calculate energy at frame boundaries vs within frames
    let num_frames = samples.len() / frame_size;
    
    let mut boundary_discontinuities = Vec::new();
    let mut within_frame_discontinuities = Vec::new();
    
    for frame_idx in 1..num_frames - 1 {
        let boundary_pos = frame_idx * frame_size;
        
        // Discontinuity at boundary
        if boundary_pos > 0 && boundary_pos < samples.len() {
            let disc = calculate_local_discontinuity(samples, boundary_pos);
            boundary_discontinuities.push(disc);
        }
        
        // Discontinuity within frame (at quarter positions)
        for offset in [frame_size / 4, frame_size / 2, 3 * frame_size / 4] {
            let pos = frame_idx * frame_size + offset;
            if pos < samples.len() {
                let disc = calculate_local_discontinuity(samples, pos);
                within_frame_discontinuities.push(disc);
            }
        }
    }
    
    if boundary_discontinuities.is_empty() || within_frame_discontinuities.is_empty() {
        return 0.0;
    }
    
    let avg_boundary = boundary_discontinuities.iter().sum::<f32>() 
        / boundary_discontinuities.len() as f32;
    let avg_within = within_frame_discontinuities.iter().sum::<f32>() 
        / within_frame_discontinuities.len() as f32;
    
    // If boundaries have significantly more discontinuity, there's a pattern
    if avg_within > 1e-10 {
        let ratio = avg_boundary / avg_within;
        (ratio - 1.0).max(0.0).min(1.0)
    } else {
        0.0
    }
}

/// Calculate local signal discontinuity
fn calculate_local_discontinuity(samples: &[f32], position: usize) -> f32 {
    let window = 16;
    
    if position < window || position + window >= samples.len() {
        return 0.0;
    }
    
    // Second derivative (acceleration) at position
    let before = &samples[position - window..position];
    let after = &samples[position..position + window];
    
    // Linear prediction from before
    let slope_before = if before.len() > 1 {
        (before[before.len() - 1] - before[0]) / before.len() as f32
    } else {
        0.0
    };
    
    let predicted = before[before.len() - 1] + slope_before;
    let actual = after[0];
    
    (predicted - actual).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_discontinuity() {
        // Smooth signal should have low discontinuity
        let smooth: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let disc = calculate_local_discontinuity(&smooth, 50);
        assert!(disc < 0.1);
    }
}
