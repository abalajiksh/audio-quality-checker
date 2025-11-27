// src/bit_depth.rs
//
// Advanced bit depth detection using multiple analysis methods.
// Detects fake 24-bit files that are actually 16-bit with zero-padded LSBs.

use crate::decoder::AudioData;
use std::collections::HashMap;

/// Comprehensive bit depth analysis result
#[derive(Debug, Clone)]
pub struct BitDepthAnalysis {
    /// Claimed bit depth from file metadata
    pub claimed_bit_depth: u32,
    /// Detected actual bit depth
    pub actual_bit_depth: u32,
    /// Confidence in detection (0.0 to 1.0)
    pub confidence: f32,
    /// Individual method results for transparency
    pub method_results: BitDepthMethodResults,
    /// Whether a mismatch was detected
    pub is_mismatch: bool,
    /// Detailed evidence for the detection
    pub evidence: Vec<String>,
}

/// Results from individual detection methods
#[derive(Debug, Clone)]
pub struct BitDepthMethodResults {
    /// LSB precision analysis result
    pub lsb_method: u32,
    /// Histogram analysis result
    pub histogram_method: u32,
    /// Quantization noise analysis result
    pub noise_method: u32,
    /// Value clustering analysis result
    pub clustering_method: u32,
}

/// Analyze bit depth using multiple methods
pub fn analyze_bit_depth(audio: &AudioData) -> BitDepthAnalysis {
    let samples = &audio.samples;
    
    if samples.is_empty() {
        return BitDepthAnalysis {
            claimed_bit_depth: audio.claimed_bit_depth,
            actual_bit_depth: audio.claimed_bit_depth,
            confidence: 0.0,
            method_results: BitDepthMethodResults {
                lsb_method: audio.claimed_bit_depth,
                histogram_method: audio.claimed_bit_depth,
                noise_method: audio.claimed_bit_depth,
                clustering_method: audio.claimed_bit_depth,
            },
            is_mismatch: false,
            evidence: vec!["No samples to analyze".to_string()],
        };
    }

    // Run all detection methods
    let lsb_result = analyze_lsb_precision(samples);
    let histogram_result = analyze_histogram(samples);
    let noise_result = analyze_quantization_noise(samples);
    let clustering_result = analyze_value_clustering(samples);
    
    // Collect results
    let results = vec![
        lsb_result.0,
        histogram_result.0,
        noise_result.0,
        clustering_result.0,
    ];
    
    // Use voting with weighted confidence
    let (actual_bit_depth, confidence) = vote_bit_depth(&results, &[
        lsb_result.1,
        histogram_result.1,
        noise_result.1,
        clustering_result.1,
    ]);
    
    // Collect evidence
    let mut evidence = Vec::new();
    evidence.push(format!("LSB analysis: {} bit (confidence: {:.1}%)", 
        lsb_result.0, lsb_result.1 * 100.0));
    evidence.push(format!("Histogram analysis: {} bit (confidence: {:.1}%)", 
        histogram_result.0, histogram_result.1 * 100.0));
    evidence.push(format!("Noise floor analysis: {} bit (confidence: {:.1}%)", 
        noise_result.0, noise_result.1 * 100.0));
    evidence.push(format!("Clustering analysis: {} bit (confidence: {:.1}%)", 
        clustering_result.0, clustering_result.1 * 100.0));
    
    let is_mismatch = actual_bit_depth < audio.claimed_bit_depth 
        && (audio.claimed_bit_depth - actual_bit_depth) >= 8
        && confidence > 0.7;
    
    if is_mismatch {
        evidence.push(format!(
            "MISMATCH: File claims {} bit but analysis indicates {} bit",
            audio.claimed_bit_depth, actual_bit_depth
        ));
    }

    BitDepthAnalysis {
        claimed_bit_depth: audio.claimed_bit_depth,
        actual_bit_depth,
        confidence,
        method_results: BitDepthMethodResults {
            lsb_method: lsb_result.0,
            histogram_method: histogram_result.0,
            noise_method: noise_result.0,
            clustering_method: clustering_result.0,
        },
        is_mismatch,
        evidence,
    }
}

/// Analyze LSB (Least Significant Bit) precision
/// True 24-bit audio has meaningful data in lower 8 bits
/// 16-bit upscaled to 24-bit has zeros or patterns in lower 8 bits
fn analyze_lsb_precision(samples: &[f32]) -> (u32, f32) {
    let test_samples = samples.len().min(100000);
    
    // Count trailing zeros when scaled to 24-bit integers
    let mut trailing_zero_counts: HashMap<u32, u32> = HashMap::new();
    let mut total_samples = 0u32;
    
    for &sample in samples.iter().take(test_samples) {
        // Skip near-zero samples (they naturally have many trailing zeros)
        if sample.abs() < 1e-6 {
            continue;
        }
        
        // Scale to 24-bit range
        let scaled = (sample * 8388607.0).round() as i32;
        
        if scaled != 0 {
            let trailing = scaled.trailing_zeros().min(24);
            *trailing_zero_counts.entry(trailing).or_insert(0) += 1;
            total_samples += 1;
        }
    }
    
    if total_samples < 1000 {
        return (16, 0.3);  // Not enough data, low confidence
    }
    
    // Calculate what percentage of samples have 8+ trailing zeros
    // (indicating they're multiples of 256, i.e., 16-bit values shifted up)
    let samples_with_8plus_zeros: u32 = trailing_zero_counts.iter()
        .filter(|(&zeros, _)| zeros >= 8)
        .map(|(_, &count)| count)
        .sum();
    
    let ratio_8plus = samples_with_8plus_zeros as f32 / total_samples as f32;
    
    // Also check for exactly 8 trailing zeros (strong 16-bit indicator)
    let samples_with_exactly_8 = *trailing_zero_counts.get(&8).unwrap_or(&0);
    let ratio_exactly_8 = samples_with_exactly_8 as f32 / total_samples as f32;
    
    // Calculate median trailing zeros
    let median_zeros = calculate_median_trailing_zeros(&trailing_zero_counts, total_samples);
    
    // Decision logic
    if ratio_8plus > 0.85 || median_zeros >= 8 {
        // Very strong evidence of 16-bit source
        let confidence = (ratio_8plus * 0.7 + 0.3).min(0.95);
        (16, confidence)
    } else if ratio_8plus > 0.5 || median_zeros >= 6 {
        // Possible 16-bit source
        (16, 0.6)
    } else if ratio_exactly_8 < 0.1 && median_zeros < 4 {
        // True 24-bit
        let confidence = (1.0 - ratio_8plus) * 0.8;
        (24, confidence)
    } else {
        // Uncertain
        (24, 0.5)
    }
}

/// Calculate median from trailing zeros histogram
fn calculate_median_trailing_zeros(counts: &HashMap<u32, u32>, total: u32) -> u32 {
    let mut cumulative = 0u32;
    let target = total / 2;
    
    for zeros in 0..=24 {
        cumulative += *counts.get(&zeros).unwrap_or(&0);
        if cumulative >= target {
            return zeros;
        }
    }
    
    0
}

/// Analyze histogram of quantized values
/// 16-bit audio has ~65536 unique values, 24-bit has ~16 million
fn analyze_histogram(samples: &[f32]) -> (u32, f32) {
    let test_samples = samples.len().min(200000);
    
    // Count unique values at 16-bit and 24-bit quantization
    let mut values_16bit: HashMap<i32, u32> = HashMap::new();
    let mut values_24bit: HashMap<i32, u32> = HashMap::new();
    
    for &sample in samples.iter().take(test_samples) {
        let q16 = (sample * 32767.0).round() as i32;
        let q24 = (sample * 8388607.0).round() as i32;
        
        *values_16bit.entry(q16).or_insert(0) += 1;
        *values_24bit.entry(q24).or_insert(0) += 1;
    }
    
    let unique_16 = values_16bit.len();
    let unique_24 = values_24bit.len();
    
    // Ratio of unique 24-bit to 16-bit values
    // True 24-bit should have significantly more unique values
    let ratio = unique_24 as f32 / unique_16.max(1) as f32;
    
    // Also check the distribution of 24-bit values
    // 16-bit upscaled will cluster on multiples of 256
    let multiples_of_256: usize = values_24bit.keys()
        .filter(|&&v| v % 256 == 0 || v % 256 == 255 || v % 256 == 1)
        .count();
    let clustering_ratio = multiples_of_256 as f32 / unique_24 as f32;
    
    // Decision logic
    if ratio < 1.5 {
        // Almost same unique values at both quantizations -> definitely 16-bit
        (16, 0.95)
    } else if ratio < 3.0 && clustering_ratio > 0.7 {
        // Low ratio with high clustering -> likely 16-bit
        (16, 0.8)
    } else if ratio > 50.0 && clustering_ratio < 0.3 {
        // Very high ratio, low clustering -> true 24-bit
        (24, 0.9)
    } else if ratio > 10.0 {
        // Moderate evidence of 24-bit
        (24, 0.7)
    } else {
        // Uncertain
        let confidence = if ratio > 5.0 { 0.6 } else { 0.5 };
        if clustering_ratio > 0.5 {
            (16, confidence)
        } else {
            (24, confidence)
        }
    }
}

/// Analyze quantization noise floor
/// 16-bit has theoretical noise floor around -96 dB
/// 24-bit has theoretical noise floor around -144 dB
fn analyze_quantization_noise(samples: &[f32]) -> (u32, f32) {
    let section_size = 16384;
    let num_sections = (samples.len() / section_size).min(20);
    
    if num_sections == 0 {
        return (16, 0.3);
    }
    
    // Find quiet sections and measure their noise floor
    let mut quiet_sections: Vec<(usize, f32)> = Vec::new();
    
    for i in 0..num_sections {
        let start = i * section_size;
        let end = (start + section_size).min(samples.len());
        let section = &samples[start..end];
        
        // Calculate RMS
        let rms = (section.iter().map(|s| s * s).sum::<f32>() / section.len() as f32).sqrt();
        
        // Skip silent sections (might be digital silence)
        if rms > 1e-8 && rms < 0.01 {  // Quiet but not silent
            quiet_sections.push((start, rms));
        }
    }
    
    if quiet_sections.is_empty() {
        // No quiet sections found - analyze overall
        return analyze_overall_noise(samples);
    }
    
    // Sort by RMS and take quietest sections
    quiet_sections.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    // Analyze LSBs in quiet sections
    let mut lsb_noise_sum = 0.0f32;
    let mut count = 0;
    
    for (start, _) in quiet_sections.iter().take(5) {
        let end = (*start + section_size).min(samples.len());
        let section = &samples[*start..end];
        
        // Look at differences between adjacent samples
        let diffs: Vec<f32> = section.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .filter(|&d| d > 1e-10 && d < 0.001)
            .collect();
        
        if diffs.len() > 100 {
            // Calculate minimum step size
            let mut sorted_diffs = diffs.clone();
            sorted_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            // 10th percentile as noise floor estimate
            let noise_step = sorted_diffs[sorted_diffs.len() / 10];
            lsb_noise_sum += noise_step;
            count += 1;
        }
    }
    
    if count == 0 {
        return (24, 0.5);
    }
    
    let avg_noise_step = lsb_noise_sum / count as f32;
    
    // Convert to bit depth estimate
    // 16-bit step: 1/32768 ≈ 3e-5
    // 24-bit step: 1/8388608 ≈ 1.2e-7
    
    let step_16bit = 1.0 / 32768.0;
    let step_24bit = 1.0 / 8388608.0;
    
    if avg_noise_step > step_16bit * 0.5 {
        (16, 0.85)
    } else if avg_noise_step < step_24bit * 10.0 {
        (24, 0.8)
    } else if avg_noise_step < step_16bit * 0.1 {
        (24, 0.7)
    } else {
        (16, 0.6)
    }
}

/// Fallback noise analysis for files without quiet sections
fn analyze_overall_noise(samples: &[f32]) -> (u32, f32) {
    // Use high-pass filtered signal to estimate noise
    let hp_samples: Vec<f32> = samples.windows(2)
        .map(|w| w[1] - w[0])
        .collect();
    
    // RMS of high-pass signal
    let hp_rms = (hp_samples.iter().map(|s| s * s).sum::<f32>() / hp_samples.len() as f32).sqrt();
    
    // Very rough estimate based on high-frequency content
    // This is less reliable than quiet section analysis
    if hp_rms < 1e-5 {
        (24, 0.5)
    } else if hp_rms > 1e-4 {
        (16, 0.5)
    } else {
        (16, 0.4)  // Default to 16-bit with low confidence
    }
}

/// Analyze value clustering patterns
/// 16-bit audio upscaled to 24-bit shows values clustering on grid
fn analyze_value_clustering(samples: &[f32]) -> (u32, f32) {
    let test_samples = samples.len().min(100000);
    
    // Look at the 8 LSBs when quantized to 24-bit
    let mut lsb_distribution: HashMap<u8, u32> = HashMap::new();
    
    for &sample in samples.iter().take(test_samples) {
        if sample.abs() < 1e-6 {
            continue;
        }
        
        let q24 = (sample * 8388607.0).round() as i32;
        let lsb_8 = (q24.abs() & 0xFF) as u8;  // Lower 8 bits
        
        *lsb_distribution.entry(lsb_8).or_insert(0) += 1;
    }
    
    if lsb_distribution.is_empty() {
        return (16, 0.3);
    }
    
    // Count how many LSB values are used
    let unique_lsb_values = lsb_distribution.len();
    
    // Check concentration at 0x00 and 0x80 (typical for 16-bit)
    let count_00 = *lsb_distribution.get(&0x00).unwrap_or(&0);
    let count_80 = *lsb_distribution.get(&0x80).unwrap_or(&0);
    let total: u32 = lsb_distribution.values().sum();
    
    let concentrated_ratio = (count_00 + count_80) as f32 / total as f32;
    
    // Calculate entropy of LSB distribution
    let entropy = calculate_entropy(&lsb_distribution);
    let max_entropy = 8.0;  // log2(256)
    let normalized_entropy = entropy / max_entropy;
    
    // Decision logic
    if unique_lsb_values < 10 || concentrated_ratio > 0.8 {
        // Very few LSB values or highly concentrated -> 16-bit
        (16, 0.9)
    } else if normalized_entropy > 0.95 && unique_lsb_values > 200 {
        // High entropy, many unique values -> true 24-bit
        (24, 0.85)
    } else if normalized_entropy < 0.5 || unique_lsb_values < 50 {
        // Low entropy or few unique values -> 16-bit
        (16, 0.75)
    } else {
        // Moderate case
        if normalized_entropy > 0.8 {
            (24, 0.6)
        } else {
            (16, 0.55)
        }
    }
}

/// Calculate Shannon entropy
fn calculate_entropy(distribution: &HashMap<u8, u32>) -> f32 {
    let total: u32 = distribution.values().sum();
    if total == 0 {
        return 0.0;
    }
    
    distribution.values()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f32 / total as f32;
            -p * p.log2()
        })
        .sum()
}

/// Vote on bit depth using weighted results
fn vote_bit_depth(results: &[u32], confidences: &[f32]) -> (u32, f32) {
    let mut vote_16 = 0.0f32;
    let mut vote_24 = 0.0f32;
    
    for (i, &result) in results.iter().enumerate() {
        let weight = confidences[i];
        if result <= 16 {
            vote_16 += weight;
        } else {
            vote_24 += weight;
        }
    }
    
    let total = vote_16 + vote_24;
    if total < 0.1 {
        return (16, 0.3);
    }
    
    if vote_16 > vote_24 {
        (16, vote_16 / total)
    } else {
        (24, vote_24 / total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_calculation() {
        // Uniform distribution should have high entropy
        let mut uniform: HashMap<u8, u32> = HashMap::new();
        for i in 0..=255 {
            uniform.insert(i, 100);
        }
        let entropy = calculate_entropy(&uniform);
        assert!(entropy > 7.9);  // Close to 8.0

        // Single value should have zero entropy
        let mut single: HashMap<u8, u32> = HashMap::new();
        single.insert(0, 1000);
        let entropy = calculate_entropy(&single);
        assert!(entropy < 0.001);
    }
}
