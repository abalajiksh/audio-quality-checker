// src/spectral.rs
//
// Advanced spectral analysis for audio quality detection.
// Multi-frame analysis, frequency cutoff detection, spectral signatures.

use crate::dsp::{
    FftProcessor, WindowType, moving_average, median, amplitude_to_db,
    spectral_flatness, spectral_rolloff, spectral_centroid,
};
use crate::decoder::AudioData;
use anyhow::Result;

/// Comprehensive spectral analysis result
#[derive(Debug, Clone)]
pub struct SpectralAnalysis {
    /// Detected frequency cutoff in Hz
    pub frequency_cutoff: f32,
    /// Confidence in the cutoff detection (0.0 to 1.0)
    pub cutoff_confidence: f32,
    /// Spectral rolloff at 95% energy
    pub rolloff_95: f32,
    /// Spectral rolloff at 85% energy
    pub rolloff_85: f32,
    /// Rolloff steepness in dB/octave
    pub rolloff_steepness: f32,
    /// Whether a "brick wall" cutoff was detected
    pub has_brick_wall: bool,
    /// Whether a pre-cutoff shelf pattern was detected
    pub has_shelf_pattern: bool,
    /// Average spectral flatness
    pub spectral_flatness: f32,
    /// Spectral centroid (brightness)
    pub spectral_centroid: f32,
    /// High frequency energy ratio (above 15kHz)
    pub hf_energy_ratio: f32,
    /// Detected spectral artifacts
    pub has_artifacts: bool,
    /// Artifact score (higher = more artifacts)
    pub artifact_score: f32,
}

/// Multi-frame spectral analyzer
pub struct SpectralAnalyzer {
    fft_processor: FftProcessor,
    fft_size: usize,
    hop_size: usize,
    sample_rate: u32,
}

impl SpectralAnalyzer {
    pub fn new(fft_size: usize, hop_size: usize, sample_rate: u32) -> Self {
        Self {
            fft_processor: FftProcessor::new(fft_size, WindowType::BlackmanHarris),
            fft_size,
            hop_size,
            sample_rate,
        }
    }

    /// Perform comprehensive spectral analysis using multiple frames
    pub fn analyze(&mut self, audio: &AudioData) -> Result<SpectralAnalysis> {
        let mono = crate::decoder::extract_mono(audio);
        
        // Analyze multiple frames spread across the track
        let num_analysis_frames = 30;
        let frame_positions = self.get_frame_positions(&mono, num_analysis_frames);
        
        let mut cutoff_estimates = Vec::new();
        let mut rolloff_95_values = Vec::new();
        let mut rolloff_85_values = Vec::new();
        let mut flatness_values = Vec::new();
        let mut centroid_values = Vec::new();
        let mut hf_ratios = Vec::new();
        let mut all_magnitudes: Vec<Vec<f32>> = Vec::new();
        
        for &pos in &frame_positions {
            if pos + self.fft_size > mono.len() {
                continue;
            }
            
            let frame = &mono[pos..pos + self.fft_size];
            let magnitudes = self.fft_processor.magnitude_spectrum(frame);
            
            // Skip very quiet frames
            let frame_energy: f32 = magnitudes.iter().map(|m| m * m).sum();
            if frame_energy < 1e-8 {
                continue;
            }
            
            cutoff_estimates.push(self.find_cutoff_derivative(&magnitudes));
            rolloff_95_values.push(spectral_rolloff(&magnitudes, self.sample_rate, 0.95));
            rolloff_85_values.push(spectral_rolloff(&magnitudes, self.sample_rate, 0.85));
            flatness_values.push(spectral_flatness(&magnitudes));
            centroid_values.push(spectral_centroid(&magnitudes, self.sample_rate));
            hf_ratios.push(self.calculate_hf_ratio(&magnitudes));
            all_magnitudes.push(magnitudes);
        }
        
        if cutoff_estimates.is_empty() {
            return Ok(SpectralAnalysis::default_for_sample_rate(self.sample_rate));
        }
        
        // Use robust statistics (median) to reject outliers
        let frequency_cutoff = median(&mut cutoff_estimates.clone());
        let rolloff_95 = median(&mut rolloff_95_values.clone());
        let rolloff_85 = median(&mut rolloff_85_values.clone());
        let avg_flatness = flatness_values.iter().sum::<f32>() / flatness_values.len() as f32;
        let avg_centroid = centroid_values.iter().sum::<f32>() / centroid_values.len() as f32;
        let avg_hf_ratio = hf_ratios.iter().sum::<f32>() / hf_ratios.len() as f32;
        
        // Calculate cutoff confidence based on consistency
        let cutoff_std = self.std_dev(&cutoff_estimates);
        let cutoff_confidence = (1.0 - (cutoff_std / frequency_cutoff).min(1.0)).max(0.0);
        
        // Analyze spectral shape
        let avg_magnitude = self.average_magnitude_spectrum(&all_magnitudes);
        let rolloff_steepness = self.calculate_rolloff_steepness(&avg_magnitude, frequency_cutoff);
        let has_brick_wall = rolloff_steepness > 30.0;
        let has_shelf_pattern = self.detect_shelf_pattern(&avg_magnitude, frequency_cutoff);
        
        // Detect artifacts
        let (has_artifacts, artifact_score) = self.detect_spectral_artifacts(&all_magnitudes);
        
        Ok(SpectralAnalysis {
            frequency_cutoff,
            cutoff_confidence,
            rolloff_95,
            rolloff_85,
            rolloff_steepness,
            has_brick_wall,
            has_shelf_pattern,
            spectral_flatness: avg_flatness,
            spectral_centroid: avg_centroid,
            hf_energy_ratio: avg_hf_ratio,
            has_artifacts,
            artifact_score,
        })
    }

    /// Get evenly-spaced frame positions, avoiding very start and end
    fn get_frame_positions(&self, samples: &[f32], num_frames: usize) -> Vec<usize> {
        let usable_length = samples.len().saturating_sub(self.fft_size);
        if usable_length == 0 {
            return vec![];
        }
        
        // Skip first and last 5% of the track
        let start = usable_length / 20;
        let end = usable_length * 19 / 20;
        let range = end - start;
        
        if range < num_frames {
            return (start..end).step_by(self.hop_size).collect();
        }
        
        (0..num_frames)
            .map(|i| start + (range * i) / num_frames)
            .collect()
    }

    /// Find frequency cutoff using derivative analysis
    /// Looks for where the spectrum "falls off a cliff"
    fn find_cutoff_derivative(&self, magnitudes: &[f32]) -> f32 {
        let db_spectrum: Vec<f32> = magnitudes.iter()
            .map(|&m| amplitude_to_db(m.max(1e-10)))
            .collect();
        
        // Smooth spectrum
        let smoothed = moving_average(&db_spectrum, 15);
        
        let nyquist = self.sample_rate as f32 / 2.0;
        let bin_hz = nyquist / magnitudes.len() as f32;
        
        // Find peak level for reference
        let peak_db = smoothed.iter().cloned().fold(f32::MIN, f32::max);
        
        // Start from high frequencies, look for where signal rises above noise
        // We're looking for the frequency where the spectrum is still "alive"
        let noise_threshold = peak_db - 50.0;  // 50 dB below peak
        
        // Search from 80% of Nyquist downward
        let search_start = (magnitudes.len() * 8) / 10;
        let search_end = magnitudes.len() / 10;
        
        for i in (search_end..search_start).rev() {
            // Check if this bin and neighbors are above noise
            let local_avg = if i > 2 && i < smoothed.len() - 2 {
                (smoothed[i-2] + smoothed[i-1] + smoothed[i] + smoothed[i+1] + smoothed[i+2]) / 5.0
            } else {
                smoothed[i]
            };
            
            if local_avg > noise_threshold {
                // Found content - check for cliff
                // Look at derivative over next octave
                let octave_bins = i / 2;
                if i + octave_bins < smoothed.len() {
                    let drop = smoothed[i] - smoothed[(i + octave_bins).min(smoothed.len() - 1)];
                    if drop > 20.0 {
                        // Sharp cliff detected
                        return i as f32 * bin_hz;
                    }
                }
                
                return i as f32 * bin_hz;
            }
        }
        
        // Fallback: return 95% rolloff
        spectral_rolloff(magnitudes, self.sample_rate, 0.95)
    }

    /// Calculate rolloff steepness in dB/octave
    fn calculate_rolloff_steepness(&self, magnitudes: &[f32], cutoff_hz: f32) -> f32 {
        let nyquist = self.sample_rate as f32 / 2.0;
        let bin_hz = nyquist / magnitudes.len() as f32;
        
        let cutoff_bin = (cutoff_hz / bin_hz) as usize;
        if cutoff_bin >= magnitudes.len() - 10 {
            return 0.0;
        }
        
        let db_spectrum: Vec<f32> = magnitudes.iter()
            .map(|&m| amplitude_to_db(m.max(1e-10)))
            .collect();
        
        // Measure dB drop over one octave above cutoff
        let octave_above_bin = (cutoff_bin * 2).min(magnitudes.len() - 1);
        
        // Average a few bins around each point
        let window = 3;
        let level_at_cutoff: f32 = db_spectrum[cutoff_bin.saturating_sub(window)..cutoff_bin + window]
            .iter().sum::<f32>() / (2 * window) as f32;
        
        let level_octave_above: f32 = db_spectrum[octave_above_bin.saturating_sub(window)..octave_above_bin + window]
            .iter().sum::<f32>() / (2 * window) as f32;
        
        // dB drop per octave
        (level_at_cutoff - level_octave_above).max(0.0)
    }

    /// Detect shelf pattern (characteristic of some AAC encodings)
    fn detect_shelf_pattern(&self, magnitudes: &[f32], cutoff_hz: f32) -> bool {
        let nyquist = self.sample_rate as f32 / 2.0;
        let bin_hz = nyquist / magnitudes.len() as f32;
        
        let cutoff_bin = (cutoff_hz / bin_hz) as usize;
        if cutoff_bin < 20 || cutoff_bin >= magnitudes.len() - 20 {
            return false;
        }
        
        let db_spectrum: Vec<f32> = magnitudes.iter()
            .map(|&m| amplitude_to_db(m.max(1e-10)))
            .collect();
        
        // Look for a "step down" pattern before the cutoff
        // Shelf patterns typically have a flat region, then a step, then another flat region
        
        let region_below = &db_spectrum[cutoff_bin - 20..cutoff_bin - 5];
        let region_above = &db_spectrum[cutoff_bin + 5..cutoff_bin + 20];
        
        let avg_below: f32 = region_below.iter().sum::<f32>() / region_below.len() as f32;
        let avg_above: f32 = region_above.iter().sum::<f32>() / region_above.len() as f32;
        
        // Check for step pattern (6-15 dB drop with low variance in both regions)
        let step_size = avg_below - avg_above;
        let var_below: f32 = region_below.iter().map(|&x| (x - avg_below).powi(2)).sum::<f32>() / region_below.len() as f32;
        let var_above: f32 = region_above.iter().map(|&x| (x - avg_above).powi(2)).sum::<f32>() / region_above.len() as f32;
        
        step_size > 6.0 && step_size < 15.0 && var_below < 10.0 && var_above < 10.0
    }

    /// Calculate high-frequency energy ratio
    fn calculate_hf_ratio(&self, magnitudes: &[f32]) -> f32 {
        let nyquist = self.sample_rate as f32 / 2.0;
        let hf_threshold = 15000.0;  // Above 15 kHz
        
        if hf_threshold >= nyquist {
            return 0.0;
        }
        
        let hf_bin = ((hf_threshold / nyquist) * magnitudes.len() as f32) as usize;
        
        let total_energy: f32 = magnitudes.iter().map(|m| m * m).sum();
        let hf_energy: f32 = magnitudes[hf_bin..].iter().map(|m| m * m).sum();
        
        if total_energy > 1e-10 {
            hf_energy / total_energy
        } else {
            0.0
        }
    }

    /// Detect spectral artifacts (gaps, notches, unusual patterns)
    fn detect_spectral_artifacts(&self, all_magnitudes: &[Vec<f32>]) -> (bool, f32) {
        if all_magnitudes.is_empty() {
            return (false, 0.0);
        }
        
        let avg_magnitude = self.average_magnitude_spectrum(all_magnitudes);
        
        // Look for notches (sudden dips in spectrum)
        let db_spectrum: Vec<f32> = avg_magnitude.iter()
            .map(|&m| amplitude_to_db(m.max(1e-10)))
            .collect();
        
        let smoothed = moving_average(&db_spectrum, 20);
        
        let mut notch_score = 0.0;
        let window = 30;
        
        // Analyze middle third of spectrum (most likely to show artifacts)
        let start = db_spectrum.len() / 3;
        let end = (db_spectrum.len() * 2) / 3;
        
        for i in (start + window)..(end - window) {
            let local_value = db_spectrum[i];
            let smoothed_value = smoothed[i];
            
            // Large negative deviation from smoothed = potential notch
            if smoothed_value - local_value > 15.0 {
                notch_score += (smoothed_value - local_value - 15.0) / 100.0;
            }
        }
        
        // Also check for unusual spectral variance between frames
        let frame_variance = self.calculate_frame_variance(all_magnitudes);
        
        let artifact_score = notch_score + frame_variance * 0.1;
        let has_artifacts = artifact_score > 0.5;
        
        (has_artifacts, artifact_score)
    }

    /// Calculate variance between frames
    fn calculate_frame_variance(&self, all_magnitudes: &[Vec<f32>]) -> f32 {
        if all_magnitudes.len() < 2 {
            return 0.0;
        }
        
        let avg = self.average_magnitude_spectrum(all_magnitudes);
        
        let variance: f32 = all_magnitudes.iter()
            .map(|frame| {
                frame.iter()
                    .zip(&avg)
                    .map(|(&f, &a)| (f - a).powi(2))
                    .sum::<f32>()
            })
            .sum::<f32>() / (all_magnitudes.len() * avg.len()) as f32;
        
        variance.sqrt()
    }

    /// Compute average magnitude spectrum across frames
    fn average_magnitude_spectrum(&self, all_magnitudes: &[Vec<f32>]) -> Vec<f32> {
        if all_magnitudes.is_empty() {
            return vec![];
        }
        
        let num_bins = all_magnitudes[0].len();
        let mut avg = vec![0.0f32; num_bins];
        
        for frame in all_magnitudes {
            for (i, &mag) in frame.iter().enumerate() {
                if i < num_bins {
                    avg[i] += mag;
                }
            }
        }
        
        let scale = 1.0 / all_magnitudes.len() as f32;
        for val in &mut avg {
            *val *= scale;
        }
        
        avg
    }

    /// Calculate standard deviation
    fn std_dev(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        
        variance.sqrt()
    }
}

impl SpectralAnalysis {
    /// Create default analysis for when processing fails
    pub fn default_for_sample_rate(sample_rate: u32) -> Self {
        let nyquist = sample_rate as f32 / 2.0;
        Self {
            frequency_cutoff: nyquist,
            cutoff_confidence: 0.0,
            rolloff_95: nyquist,
            rolloff_85: nyquist * 0.9,
            rolloff_steepness: 0.0,
            has_brick_wall: false,
            has_shelf_pattern: false,
            spectral_flatness: 0.0,
            spectral_centroid: 0.0,
            hf_energy_ratio: 0.0,
            has_artifacts: false,
            artifact_score: 0.0,
        }
    }
}

/// Spectral signature for codec identification
#[derive(Debug, Clone)]
pub struct SpectralSignature {
    pub cutoff_range: (f32, f32),
    pub typical_rolloff_steepness: f32,
    pub has_brick_wall: bool,
    pub has_shelf: bool,
}

/// Known encoder signatures for comparison
pub fn get_encoder_signatures() -> Vec<(&'static str, SpectralSignature)> {
    vec![
        ("MP3 128kbps", SpectralSignature {
            cutoff_range: (15500.0, 16500.0),
            typical_rolloff_steepness: 40.0,
            has_brick_wall: true,
            has_shelf: false,
        }),
        ("MP3 192kbps", SpectralSignature {
            cutoff_range: (17500.0, 18500.0),
            typical_rolloff_steepness: 35.0,
            has_brick_wall: true,
            has_shelf: false,
        }),
        ("MP3 256kbps", SpectralSignature {
            cutoff_range: (19000.0, 20000.0),
            typical_rolloff_steepness: 30.0,
            has_brick_wall: true,
            has_shelf: false,
        }),
        ("MP3 320kbps", SpectralSignature {
            cutoff_range: (20000.0, 20500.0),
            typical_rolloff_steepness: 25.0,
            has_brick_wall: true,
            has_shelf: false,
        }),
        ("AAC 128kbps", SpectralSignature {
            cutoff_range: (15000.0, 16000.0),
            typical_rolloff_steepness: 20.0,
            has_brick_wall: false,
            has_shelf: true,
        }),
        ("AAC 256kbps", SpectralSignature {
            cutoff_range: (18000.0, 20000.0),
            typical_rolloff_steepness: 15.0,
            has_brick_wall: false,
            has_shelf: true,
        }),
        ("Vorbis q5", SpectralSignature {
            cutoff_range: (16000.0, 17500.0),
            typical_rolloff_steepness: 20.0,
            has_brick_wall: false,
            has_shelf: false,
        }),
        ("Vorbis q8", SpectralSignature {
            cutoff_range: (19000.0, 20500.0),
            typical_rolloff_steepness: 15.0,
            has_brick_wall: false,
            has_shelf: false,
        }),
        ("Opus 64kbps", SpectralSignature {
            cutoff_range: (11500.0, 12500.0),
            typical_rolloff_steepness: 30.0,
            has_brick_wall: true,
            has_shelf: false,
        }),
        ("Opus 128kbps", SpectralSignature {
            cutoff_range: (19000.0, 20500.0),
            typical_rolloff_steepness: 20.0,
            has_brick_wall: true,
            has_shelf: false,
        }),
    ]
}

/// Match analysis against known signatures
pub fn match_signature(analysis: &SpectralAnalysis) -> Option<(&'static str, f32)> {
    let signatures = get_encoder_signatures();
    let mut best_match: Option<(&str, f32)> = None;
    
    for (name, sig) in &signatures {
        let mut score = 0.0f32;
        let mut factors = 0.0f32;
        
        // Cutoff range match
        if analysis.frequency_cutoff >= sig.cutoff_range.0 
            && analysis.frequency_cutoff <= sig.cutoff_range.1 {
            score += 40.0;
        } else {
            let distance = if analysis.frequency_cutoff < sig.cutoff_range.0 {
                sig.cutoff_range.0 - analysis.frequency_cutoff
            } else {
                analysis.frequency_cutoff - sig.cutoff_range.1
            };
            score += (40.0 - distance / 50.0).max(0.0);
        }
        factors += 40.0;
        
        // Brick wall match
        if analysis.has_brick_wall == sig.has_brick_wall {
            score += 20.0;
        }
        factors += 20.0;
        
        // Shelf pattern match
        if analysis.has_shelf_pattern == sig.has_shelf {
            score += 15.0;
        }
        factors += 15.0;
        
        // Rolloff steepness similarity
        let steepness_diff = (analysis.rolloff_steepness - sig.typical_rolloff_steepness).abs();
        score += (25.0 - steepness_diff).max(0.0);
        factors += 25.0;
        
        let confidence = score / factors;
        
        if confidence > 0.6 {
            if best_match.is_none() || confidence > best_match.unwrap().1 {
                best_match = Some((name, confidence));
            }
        }
    }
    
    best_match
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_analysis_creation() {
        let analysis = SpectralAnalysis::default_for_sample_rate(44100);
        assert!((analysis.frequency_cutoff - 22050.0).abs() < 1.0);
    }
}
