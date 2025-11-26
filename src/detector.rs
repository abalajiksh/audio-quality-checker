use anyhow::Result;
use rustfft::{FftPlanner, num_complex::Complex};
use crate::decoder::AudioData;
use std::collections::{HashMap, HashSet};


#[derive(Debug)]
pub enum DefectType {
    Mp3Transcode { cutoff_hz: u32 },
    OggVorbisTranscode { cutoff_hz: u32 },
    AacTranscode { cutoff_hz: u32 },
    OpusTranscode { cutoff_hz: u32, mode: String },
    BitDepthMismatch { claimed: u32, actual: u32 },
    Upsampled { from: u32, to: u32 },
    SpectralArtifacts,
    #[allow(dead_code)]
    LowQuality,
}


#[derive(Debug)]
pub struct QualityReport {
    pub sample_rate: u32,
    pub channels: usize,
    pub claimed_bit_depth: u32,
    pub actual_bit_depth: u32,
    pub duration_secs: f64,
    pub frequency_cutoff: f32,
    pub dynamic_range: f32,
    pub noise_floor: f32,
    pub peak_amplitude: f32,
    pub spectral_rolloff: f32,
    pub defects: Vec<DefectType>,
}


pub fn detect_quality_issues(
    audio: &AudioData,
    _expected_bit_depth: u32,
    check_upsampling: bool,
) -> Result<QualityReport> {
    let mut defects = Vec::new();

    // Analyze frequency spectrum
    let (cutoff, rolloff, has_artifacts) = analyze_frequency_spectrum(audio)?;
    
    // Detect transcodes
    let nyquist = audio.sample_rate as f32 / 2.0;
    let cutoff_ratio = cutoff / nyquist;
    
    // Only flag if cutoff is significantly below expected (< 80% of Nyquist)
    if cutoff_ratio < 0.80 {
        // For 96kHz files, MP3 cutoff will be around 16-20kHz (out of 48kHz Nyquist)
        // So cutoff_ratio will be 16000/48000 = 0.33 to 0.42
        
        if cutoff >= 7500.0 && cutoff <= 8500.0 {
            defects.push(DefectType::OpusTranscode {
                cutoff_hz: cutoff as u32,
                mode: "Wideband (8kHz)".to_string()
            });
        } else if cutoff >= 11500.0 && cutoff <= 12500.0 {
            defects.push(DefectType::OpusTranscode {
                cutoff_hz: cutoff as u32,
                mode: "Super-wideband (12kHz)".to_string()
            });
        } else if cutoff >= 14500.0 && cutoff <= 20500.0 {
            // Expanded range to catch more MP3/Vorbis transcodes
            if cutoff < 16500.0 {
                defects.push(DefectType::Mp3Transcode { cutoff_hz: cutoff as u32 });
            } else if cutoff < 18000.0 {
                defects.push(DefectType::OggVorbisTranscode { cutoff_hz: cutoff as u32 });
            } else if cutoff < 19500.0 {
                defects.push(DefectType::AacTranscode { cutoff_hz: cutoff as u32 });
            } else {
                // 19.5-20.5 kHz could be high bitrate MP3 or AAC
                defects.push(DefectType::Mp3Transcode { cutoff_hz: cutoff as u32 });
            }
        }
        
        // Add spectral artifacts if detected in low cutoff scenarios
        if has_artifacts {
            defects.push(DefectType::SpectralArtifacts);
        }
    } else if cutoff_ratio >= 0.80 && cutoff_ratio < 0.95 && has_artifacts {
        // High cutoff but with artifacts - likely high bitrate lossy
        defects.push(DefectType::SpectralArtifacts);
    }

    // Detect ACTUAL bit depth from samples
    let actual_bit_depth = detect_actual_bit_depth(audio);
    let (dynamic_range, noise_floor, peak_amp) = calculate_simple_dynamic_range(audio);

    // Flag bit depth mismatch if significant (8-bit difference minimum)
    if actual_bit_depth <= 16 && audio.bit_depth >= 24 && (audio.bit_depth - actual_bit_depth) >= 8 {
        defects.push(DefectType::BitDepthMismatch {
            claimed: audio.bit_depth,
            actual: actual_bit_depth,
        });
    }

    // Check for upsampling
    if check_upsampling {
        if let Some(original_rate) = detect_upsampling(audio, cutoff, cutoff_ratio) {
            defects.push(DefectType::Upsampled {
                from: original_rate,
                to: audio.sample_rate,
            });
        }
    }

    Ok(QualityReport {
        sample_rate: audio.sample_rate,
        channels: audio.channels,
        claimed_bit_depth: audio.bit_depth,
        actual_bit_depth,
        duration_secs: audio.duration_secs,
        frequency_cutoff: cutoff,
        dynamic_range,
        noise_floor,
        peak_amplitude: peak_amp,
        spectral_rolloff: rolloff,
        defects,
    })
}


fn detect_actual_bit_depth(audio: &AudioData) -> u32 {
    let samples = &audio.samples;
    
    if samples.is_empty() {
        return audio.bit_depth;
    }
    
    // Use three independent methods
    let lsb_depth = analyze_lsb_precision(samples);
    let quantization_depth = analyze_quantization_noise(audio);
    let distribution_depth = analyze_value_distribution(samples);
    
    // Take the median of the three methods for robustness
    let mut estimates = vec![lsb_depth, quantization_depth, distribution_depth];
    estimates.sort();
    estimates[1]  // Return median
}


fn analyze_lsb_precision(samples: &[f32]) -> u32 {
    // Analyze least significant bit patterns to determine actual precision
    let mut bit_patterns: HashMap<u32, u32> = HashMap::new();
    let test_samples = samples.len().min(50000);
    
    for &sample in samples.iter().take(test_samples) {
        if sample.abs() < 1e-10 {
            continue;  // Skip near-zero samples
        }
        
        // Scale to 24-bit integer range
        let scaled = (sample * 8388607.0) as i32;
        
        if scaled != 0 {
            let trailing_zeros = scaled.trailing_zeros();
            *bit_patterns.entry(trailing_zeros).or_insert(0) += 1;
        }
    }
    
    if bit_patterns.is_empty() {
        return 16;
    }
    
    // Find the median trailing zeros
    let total: u32 = bit_patterns.values().sum();
    let mut cumulative = 0u32;
    let mut sorted: Vec<_> = bit_patterns.iter().collect();
    sorted.sort_by_key(|(zeros, _)| *zeros);
    
    let mut median_zeros = 0u32;
    for (zeros, count) in sorted {
        cumulative += count;
        if cumulative >= total / 2 {
            median_zeros = *zeros;
            break;
        }
    }
    
    // Calculate effective bits
    let effective_bits = 24 - median_zeros;
    
    if effective_bits >= 20 {
        24
    } else if effective_bits >= 14 {
        16
    } else {
        8
    }
}


fn analyze_quantization_noise(audio: &AudioData) -> u32 {
    let samples = &audio.samples;
    let section_size = 8192;
    let num_sections = samples.len() / section_size;
    
    if num_sections == 0 {
        return audio.bit_depth;
    }
    
    // Find the quietest section
    let mut min_rms = f32::MAX;
    let mut quietest_start = 0;
    
    for i in 0..num_sections {
        let start = i * section_size;
        let end = (start + section_size).min(samples.len());
        let section = &samples[start..end];
        
        let rms: f32 = section.iter()
            .map(|s| s * s)
            .sum::<f32>() / section.len() as f32;
        
        if rms < min_rms && rms > 1e-10 {
            min_rms = rms;
            quietest_start = start;
        }
    }
    
    // Analyze quantization noise in quiet section
    let end = (quietest_start + section_size).min(samples.len());
    let quiet_section = &samples[quietest_start..end];
    
    let mut diffs: Vec<f32> = quiet_section.windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .filter(|&d| d > 1e-10)
        .collect();
    
    if diffs.is_empty() {
        return audio.bit_depth;
    }
    
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_diff = diffs[diffs.len() / 2];
    
    let noise_db = if median_diff > 1e-10 {
        20.0 * median_diff.log10()
    } else {
        -120.0
    };
    
    // Classify based on noise floor
    if noise_db < -115.0 {
        24
    } else if noise_db < -75.0 {
        16
    } else {
        8
    }
}


fn analyze_value_distribution(samples: &[f32]) -> u32 {
    let test_size = samples.len().min(100000);
    
    let mut values_16bit: HashSet<i16> = HashSet::new();
    let mut values_24bit: HashSet<i32> = HashSet::new();
    
    for &sample in samples.iter().take(test_size) {
        let q16 = (sample * 32767.0).round() as i16;
        values_16bit.insert(q16);
        
        let q24 = (sample * 8388607.0).round() as i32;
        values_24bit.insert(q24);
    }
    
    let unique_16 = values_16bit.len();
    let unique_24 = values_24bit.len();
    
    let ratio = unique_24 as f32 / unique_16.max(1) as f32;
    
    if ratio > 10.0 {
        24
    } else if ratio > 2.0 {
        16
    } else {
        16
    }
}


fn calculate_simple_dynamic_range(audio: &AudioData) -> (f32, f32, f32) {
    let samples = &audio.samples;
    
    if samples.is_empty() {
        return (0.0, -120.0, -120.0);
    }
    
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let peak_db = if peak > 0.0 { 20.0 * peak.log10() } else { -120.0 };
    
    let rms: f32 = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    let rms_db = if rms > 0.0 { 20.0 * rms.log10() } else { -120.0 };
    
    let dynamic_range = (peak_db - rms_db).max(0.0);
    
    (dynamic_range, rms_db, peak_db)
}


fn analyze_frequency_spectrum(audio: &AudioData) -> Result<(f32, f32, bool)> {
    let fft_size = 8192;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    // Take middle section of audio
    let start = audio.samples.len() / 2;
    let end = (start + fft_size * audio.channels).min(audio.samples.len());
    
    if end - start < fft_size {
        return Ok((audio.sample_rate as f32 / 2.0, audio.sample_rate as f32 / 2.0, false));
    }

    // Extract mono channel
    let mut signal: Vec<Complex<f32>> = audio.samples[start..end]
        .chunks(audio.channels)
        .take(fft_size)
        .map(|chunk| Complex::new(chunk[0], 0.0))
        .collect();

    // Apply Hann window
    for (i, sample) in signal.iter_mut().enumerate() {
        let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos());
        *sample *= window;
    }

    fft.process(&mut signal);

    // Calculate magnitude spectrum
    let magnitudes: Vec<f32> = signal[..fft_size / 2]
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect();

    let cutoff = find_frequency_cutoff(&magnitudes, audio.sample_rate);
    let rolloff = find_spectral_rolloff(&magnitudes, audio.sample_rate);
    let has_artifacts = detect_spectral_artifacts(&magnitudes);

    Ok((cutoff, rolloff, has_artifacts))
}


fn find_frequency_cutoff(magnitudes: &[f32], sample_rate: u32) -> f32 {
    if magnitudes.is_empty() {
        return sample_rate as f32 / 2.0;
    }
    
    let peak = magnitudes.iter().cloned().fold(0.0f32, f32::max);
    
    if peak < 1e-10 {
        return sample_rate as f32 / 2.0;
    }
    
    // Look for where magnitude drops below 1% of peak
    let threshold = peak * 0.01;
    
    // Start from high frequencies and work down
    for i in (magnitudes.len() / 2..magnitudes.len()).rev() {
        if magnitudes[i] > threshold {
            return i as f32 * sample_rate as f32 / (2.0 * magnitudes.len() as f32);
        }
    }
    
    sample_rate as f32 / 2.0
}


fn find_spectral_rolloff(magnitudes: &[f32], sample_rate: u32) -> f32 {
    let total_energy: f32 = magnitudes.iter().map(|m| m * m).sum();
    
    if total_energy < 1e-10 {
        return sample_rate as f32 / 2.0;
    }
    
    let threshold = total_energy * 0.95;
    let mut cumulative = 0.0;
    
    for (i, &mag) in magnitudes.iter().enumerate() {
        cumulative += mag * mag;
        if cumulative >= threshold {
            return i as f32 * sample_rate as f32 / (2.0 * magnitudes.len() as f32);
        }
    }
    
    sample_rate as f32 / 2.0
}


fn detect_spectral_artifacts(magnitudes: &[f32]) -> bool {
    if magnitudes.len() < 100 {
        return false;
    }
    
    let mut artifact_score = 0;
    let window = 30;
    
    let start = magnitudes.len() / 3;
    let end = magnitudes.len() * 2 / 3;
    
    for i in (start + window)..(end - window) {
        let before: f32 = magnitudes[i - window..i].iter().sum::<f32>() / window as f32;
        let after: f32 = magnitudes[i..i + window].iter().sum::<f32>() / window as f32;
        let current = magnitudes[i];
        
        let avg = (before + after) / 2.0;
        
        if avg > 0.0 && current < avg * 0.15 {
            artifact_score += 1;
        }
    }
    
    artifact_score > 50
}


fn detect_upsampling(audio: &AudioData, cutoff_freq: f32, _cutoff_ratio: f32) -> Option<u32> {
    let sample_rate = audio.sample_rate;
    
    let rate_pairs = vec![
        (44100, 88200),
        (44100, 96000),
        (44100, 176400),
        (44100, 192000),
        (48000, 96000),
        (48000, 192000),
        (96000, 176400),
        (96000, 192000),
    ];
    
    for (original, upsampled) in rate_pairs {
        if sample_rate == upsampled {
            let original_nyquist = original as f32 / 2.0;
            let diff_ratio = (cutoff_freq - original_nyquist).abs() / original_nyquist;
            
            // More lenient threshold for upsampling detection
            if diff_ratio < 0.10 {
                return Some(original);
            }
        }
    }
    
    None
}
