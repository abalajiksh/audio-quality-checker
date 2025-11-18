// src/detector.rs
use anyhow::Result;
use rustfft::{FftPlanner, num_complex::Complex};
use crate::decoder::AudioData;

#[derive(Debug)]
pub enum DefectType {
    Mp3Transcode { cutoff_hz: u32 },
    OggVorbisTranscode { cutoff_hz: u32 },
    AacTranscode { cutoff_hz: u32 },
    OpusTranscode { cutoff_hz: u32, mode: String },
    BitDepthMismatch { claimed: u32, actual: u32 },
    Upsampled { from: u32, to: u32 },
    SpectralArtifacts,
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
    expected_bit_depth: u32,
    check_upsampling: bool,
) -> Result<QualityReport> {
    let mut defects = Vec::new();

    // Analyze frequency spectrum
    let (cutoff, rolloff, has_artifacts) = analyze_frequency_spectrum(audio)?;
    
    let nyquist = audio.sample_rate as f32 / 2.0;
    let cutoff_ratio = cutoff / nyquist;

    if cutoff_ratio < 0.85 {  // Only flag if cutoff is below 85% of Nyquist
    if cutoff < 8500.0 && cutoff > 7500.0 {
        defects.push(DefectType::OpusTranscode { 
            cutoff_hz: cutoff as u32,
            mode: "Wideband (8kHz)".to_string()
        });
    } else if cutoff < 12500.0 && cutoff > 11500.0 {
        defects.push(DefectType::OpusTranscode { 
            cutoff_hz: cutoff as u32,
            mode: "Super-wideband (12kHz)".to_string()
        });
    } else if cutoff < 16500.0 && cutoff > 14500.0 {
        // MP3 range
        if cutoff < 15500.0 {
            defects.push(DefectType::Mp3Transcode { cutoff_hz: cutoff as u32 });
        } else if cutoff < 16000.0 {
            defects.push(DefectType::OggVorbisTranscode { cutoff_hz: cutoff as u32 });
        }
    } else if cutoff < 18500.0 && cutoff > 16500.0 {
        defects.push(DefectType::AacTranscode { cutoff_hz: cutoff as u32 });
    }
}

    if has_artifacts {
        defects.push(DefectType::SpectralArtifacts);
    }

    // Analyze dynamic range and bit depth
    let (dynamic_range, noise_floor, peak_amp) = analyze_dynamic_range(audio);
    let actual_bit_depth = estimate_bit_depth(dynamic_range);

    if actual_bit_depth < expected_bit_depth {
        defects.push(DefectType::BitDepthMismatch {
            claimed: audio.bit_depth,
            actual: actual_bit_depth,
        });
    }

    // Check for upsampling
    if check_upsampling {
        if let Some(original_rate) = detect_upsampling(audio, cutoff) {
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

fn analyze_frequency_spectrum(audio: &AudioData) -> Result<(f32, f32, bool)> {
    let fft_size = 8192;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    // Take middle section of audio to avoid edge effects
    let start = audio.samples.len() / 2;
    let end = (start + fft_size * audio.channels).min(audio.samples.len());
    
    if end - start < fft_size {
        return Ok((audio.sample_rate as f32 / 2.0, audio.sample_rate as f32 / 2.0, false));
    }

    // Extract mono channel for analysis
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

    // Find frequency cutoff (where energy drops significantly)
    let cutoff = find_frequency_cutoff(&magnitudes, audio.sample_rate);
    
    // Find spectral rolloff (95% energy point)
    let rolloff = find_spectral_rolloff(&magnitudes, audio.sample_rate);
    
    // Detect artifacts (irregularities in spectrum)
    let has_artifacts = detect_spectral_artifacts(&magnitudes);

    Ok((cutoff, rolloff, has_artifacts))
}

fn find_frequency_cutoff(magnitudes: &[f32], sample_rate: u32) -> f32 {
    // Use a more sophisticated approach: find where energy drops significantly
    // Average the last 10% of spectrum to determine if there's real high-frequency content
    
    let high_freq_start = (magnitudes.len() as f32 * 0.7) as usize;
    let high_freq_avg: f32 = magnitudes[high_freq_start..]
        .iter()
        .sum::<f32>() / (magnitudes.len() - high_freq_start) as f32;
    
    let mid_freq_start = (magnitudes.len() as f32 * 0.3) as usize;
    let mid_freq_end = (magnitudes.len() as f32 * 0.6) as usize;
    let mid_freq_avg: f32 = magnitudes[mid_freq_start..mid_freq_end]
        .iter()
        .sum::<f32>() / (mid_freq_end - mid_freq_start) as f32;
    
    // If high frequencies have significantly less energy than mid frequencies, find cutoff
    if mid_freq_avg > 0.0 && high_freq_avg / mid_freq_avg < 0.05 {
        // Find where energy drops to 10% of peak (more reasonable threshold)
        let peak = magnitudes.iter().cloned().fold(0.0f32, f32::max);
        let threshold = peak * 0.1;
        
        // Search from 85% of Nyquist downward
        let start_search = (magnitudes.len() as f32 * 0.85) as usize;
        
        for i in (0..start_search).rev() {
            if magnitudes[i] > threshold {
                let freq = i as f32 * sample_rate as f32 / (2.0 * magnitudes.len() as f32);
                return freq;
            }
        }
    }
    
    // Return Nyquist if no significant cutoff detected
    sample_rate as f32 / 2.0
}


fn find_spectral_rolloff(magnitudes: &[f32], sample_rate: u32) -> f32 {
    let total_energy: f32 = magnitudes.iter().map(|m| m * m).sum();
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
    // Look for more significant anomalies
    let mut artifact_count = 0;
    let window_size = 20;  // Increased from 10
    
    // Only check middle to high frequencies (where artifacts are more visible)
    let start_check = magnitudes.len() / 4;
    let end_check = magnitudes.len() * 3 / 4;
    
    for i in (start_check + window_size)..(end_check - window_size) {
        let before: f32 = magnitudes[i - window_size..i].iter().sum::<f32>() / window_size as f32;
        let current = magnitudes[i];
        let after: f32 = magnitudes[i + 1..i + window_size + 1].iter().sum::<f32>() / window_size as f32;
        
        let avg = (before + after) / 2.0;
        
        // More strict threshold: 80% drop instead of 70%
        if avg > 0.0 && current < avg * 0.2 {
            artifact_count += 1;
        }
    }
    
    // Require more artifacts to trigger (20 instead of 5)
    artifact_count > 20
}


fn analyze_dynamic_range(audio: &AudioData) -> (f32, f32, f32) {
    let samples = &audio.samples;
    
    // Find peak amplitude (RMS-based, more accurate than absolute peak)
    let window_size = 2048;
    let mut max_rms = 0.0f32;
    
    for chunk in samples.chunks(window_size) {
        let rms: f32 = chunk.iter()
            .map(|s| s * s)
            .sum::<f32>() / chunk.len() as f32;
        max_rms = max_rms.max(rms.sqrt());
    }
    
    let peak_db = if max_rms > 0.0 { 20.0 * max_rms.log10() } else { -120.0 };
    
    // Estimate noise floor using histogram method (more reliable)
    let mut amplitude_hist = vec![0u32; 1000];
    let hist_max = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    
    if hist_max > 0.0 {
        for &sample in samples {
            let normalized = (sample.abs() / hist_max * 999.0) as usize;
            if normalized < 1000 {
                amplitude_hist[normalized] += 1;
            }
        }
        
        // Find the lowest non-zero bin with significant content (>0.1% of samples)
        let threshold_count = (samples.len() as f32 * 0.001) as u32;
        let mut noise_bin = 0;
        
        for (i, &count) in amplitude_hist.iter().enumerate() {
            if count > threshold_count && i > 0 {
                noise_bin = i;
                break;
            }
        }
        
        let noise_floor = (noise_bin as f32 / 1000.0) * hist_max;
        let noise_db = if noise_floor > 0.0 { 20.0 * noise_floor.log10() } else { -120.0 };
        
        // Clamp to reasonable values
        let dynamic_range = (peak_db - noise_db).max(0.0).min(160.0);
        
        (dynamic_range, noise_db, peak_db)
    } else {
        (0.0, -120.0, -120.0)
    }
}

fn estimate_bit_depth(dynamic_range: f32) -> u32 {
    // More lenient thresholds accounting for real-world audio
    // Real music rarely uses full theoretical dynamic range
    if dynamic_range > 110.0 {
        24  // Theoretical 24-bit is 144 dB, but real audio rarely exceeds 110-120 dB
    } else if dynamic_range > 70.0 {
        16  // Theoretical 16-bit is 96 dB, but 70-90 dB is typical for real music
    } else if dynamic_range > 40.0 {
        8   // 8-bit is 48 dB theoretical
    } else {
        8
    }
}


fn detect_upsampling(audio: &AudioData, cutoff_freq: f32) -> Option<u32> {
    let sample_rate = audio.sample_rate;
    let nyquist = sample_rate as f32 / 2.0;
    
    // Common sample rate pairs
    let rate_pairs = vec![
        (44100, 88200),
        (44100, 96000),
        (44100, 176400),
        (44100, 192000),
        (48000, 96000),
        (48000, 192000),
        (96000, 192000),
    ];
    
    for (original, upsampled) in rate_pairs {
        if sample_rate == upsampled {
            let original_nyquist = original as f32 / 2.0;
            // If cutoff is near the original Nyquist, likely upsampled
            if cutoff_freq < original_nyquist * 1.1 {
                return Some(original);
            }
        }
    }
    
    None
}


// Add new function to src/detector.rs to detect Opus-specific artifacts
fn detect_opus_artifacts(audio: &AudioData) -> Result<bool> {
    let fft_size = 8192;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    // Analyze multiple sections
    let num_sections = 5;
    let section_size = audio.samples.len() / (num_sections * audio.channels);
    let mut opus_indicators = 0;

    for section in 0..num_sections {
        let start = section * section_size * audio.channels;
        let end = (start + fft_size * audio.channels).min(audio.samples.len());
        
        if end - start < fft_size * audio.channels {
            continue;
        }

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

        let magnitudes: Vec<f32> = signal[..fft_size / 2]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        // Opus characteristics:
        // 1. Spectral folding creates mirror patterns
        // 2. Band extension shows smoothed/synthetic high frequencies
        // 3. Distinct energy distribution at bandwidth boundaries
        
        if detect_spectral_folding(&magnitudes, audio.sample_rate) {
            opus_indicators += 1;
        }
        
        if detect_bandwidth_boundary(&magnitudes, audio.sample_rate) {
            opus_indicators += 1;
        }
    }

    // If multiple sections show Opus characteristics, likely transcoded
    Ok(opus_indicators > num_sections / 2)
}

fn detect_spectral_folding(magnitudes: &[f32], sample_rate: u32) -> bool {
    // Opus uses spectral folding for bandwidth extension
    // Look for correlation between low and high frequency bands
    let nyquist_idx = magnitudes.len() - 1;
    let mid_idx = magnitudes.len() / 2;
    
    // Check for suspicious similarity in spectral envelope
    let low_band: Vec<f32> = magnitudes[mid_idx/2..mid_idx].to_vec();
    let high_band: Vec<f32> = magnitudes[mid_idx..mid_idx + low_band.len()].to_vec();
    
    // Calculate correlation
    let correlation = calculate_correlation(&low_band, &high_band);
    
    // High correlation suggests folding (Opus artifact)
    correlation > 0.6
}

fn detect_bandwidth_boundary(magnitudes: &[f32], sample_rate: u32) -> bool {
    // Opus has distinct energy drops at bandwidth boundaries
    // Check for sharp transitions at 8kHz, 12kHz, or 20kHz
    let boundaries = vec![8000.0, 12000.0, 20000.0];
    
    for boundary in boundaries {
        let bin = (boundary * magnitudes.len() as f32 / (sample_rate as f32 / 2.0)) as usize;
        
        if bin >= 5 && bin < magnitudes.len() - 5 {
            let before: f32 = magnitudes[bin-5..bin].iter().sum::<f32>() / 5.0;
            let after: f32 = magnitudes[bin..bin+5].iter().sum::<f32>() / 5.0;
            
            // Sharp energy drop characteristic of Opus bandwidth boundary
            if before > 0.0 && after / before < 0.3 {
                return true;
            }
        }
    }
    
    false
}

fn calculate_correlation(signal1: &[f32], signal2: &[f32]) -> f32 {
    let len = signal1.len().min(signal2.len());
    if len == 0 {
        return 0.0;
    }
    
    let mean1: f32 = signal1[..len].iter().sum::<f32>() / len as f32;
    let mean2: f32 = signal2[..len].iter().sum::<f32>() / len as f32;
    
    let mut numerator = 0.0;
    let mut denom1 = 0.0;
    let mut denom2 = 0.0;
    
    for i in 0..len {
        let diff1 = signal1[i] - mean1;
        let diff2 = signal2[i] - mean2;
        numerator += diff1 * diff2;
        denom1 += diff1 * diff1;
        denom2 += diff2 * diff2;
    }
    
    let denominator = (denom1 * denom2).sqrt();
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}
