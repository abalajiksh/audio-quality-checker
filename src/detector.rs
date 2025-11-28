// src/detector.rs
//
// Main detection logic combining all analysis modules.
// Produces comprehensive quality reports with confidence scores.
//
// CORRECTED v2: Fixed false positives on high sample rate files (88.2kHz+)
// by using absolute frequency thresholds instead of ratio-based detection.

use anyhow::Result;
use crate::decoder::AudioData;
use crate::spectral::{SpectralAnalyzer, SpectralAnalysis, match_signature};
use crate::bit_depth::{analyze_bit_depth, BitDepthAnalysis};
use crate::stereo::{analyze_stereo, StereoAnalysis};
use crate::transients::{analyze_pre_echo, PreEchoAnalysis};
use crate::phase::{analyze_phase, PhaseAnalysis};
use crate::upsampling::{analyze_upsampling, UpsamplingAnalysis};
use crate::true_peak::{analyze_true_peak, TruePeakAnalysis};

/// Defect types that can be detected
#[derive(Debug, Clone)]
pub enum DefectType {
    /// MP3 transcode detected
    Mp3Transcode { 
        cutoff_hz: u32,
        estimated_bitrate: Option<u32>,
    },
    /// Ogg Vorbis transcode detected
    OggVorbisTranscode { 
        cutoff_hz: u32,
        estimated_quality: Option<u32>,
    },
    /// AAC transcode detected  
    AacTranscode { 
        cutoff_hz: u32,
        estimated_bitrate: Option<u32>,
    },
    /// Opus transcode detected
    OpusTranscode { 
        cutoff_hz: u32, 
        mode: String,
    },
    /// Bit depth mismatch (e.g., 16-bit labeled as 24-bit)
    BitDepthMismatch { 
        claimed: u32, 
        actual: u32,
        confidence: f32,
    },
    /// Upsampled from lower sample rate
    Upsampled { 
        from: u32, 
        to: u32,
        confidence: f32,
    },
    /// Spectral artifacts detected
    SpectralArtifacts {
        artifact_score: f32,
    },
    /// Joint stereo encoding detected
    JointStereo {
        confidence: f32,
    },
    /// Pre-echo artifacts (characteristic of transform codecs)
    PreEcho {
        score: f32,
    },
    /// Phase discontinuities detected
    PhaseDiscontinuities {
        score: f32,
    },
    /// Clipping detected
    Clipping {
        percentage: f32,
    },
    /// Inter-sample overs (true peak > 0 dBFS)
    InterSampleOvers {
        count: usize,
        max_level_db: f32,
    },
    /// Low quality encoding
    LowQuality {
        description: String,
    },
}

/// Defect with confidence score
#[derive(Debug, Clone)]
pub struct DetectedDefect {
    pub defect_type: DefectType,
    pub confidence: f32,
    pub evidence: Vec<String>,
}

/// Comprehensive quality report
#[derive(Debug)]
pub struct QualityReport {
    // Basic file info
    pub sample_rate: u32,
    pub channels: usize,
    pub claimed_bit_depth: u32,
    pub actual_bit_depth: u32,
    pub duration_secs: f64,
    pub codec_name: String,
    
    // Spectral analysis
    pub frequency_cutoff: f32,
    pub spectral_rolloff: f32,
    pub rolloff_steepness: f32,
    pub has_brick_wall: bool,
    pub spectral_flatness: f32,
    
    // Dynamic analysis
    pub dynamic_range: f32,
    pub noise_floor: f32,
    pub peak_amplitude: f32,
    pub true_peak: f32,
    pub crest_factor: f32,
    
    // Stereo analysis (if applicable)
    pub stereo_width: Option<f32>,
    pub channel_correlation: Option<f32>,
    
    // Detected defects
    pub defects: Vec<DetectedDefect>,
    
    // Overall assessment
    pub quality_score: f32,  // 0.0 = definitely transcoded, 1.0 = likely lossless
    pub is_likely_lossless: bool,
    
    // Detailed analysis results (for verbose output)
    pub spectral_analysis: SpectralAnalysis,
    pub bit_depth_analysis: BitDepthAnalysis,
    pub stereo_analysis: Option<StereoAnalysis>,
    pub pre_echo_analysis: PreEchoAnalysis,
    pub phase_analysis: PhaseAnalysis,
    pub upsampling_analysis: UpsamplingAnalysis,
    pub true_peak_analysis: TruePeakAnalysis,
}

/// Detection configuration options
#[derive(Debug, Clone)]
pub struct DetectionConfig {
    /// Expected bit depth (for comparison)
    pub expected_bit_depth: u32,
    /// Check for upsampling
    pub check_upsampling: bool,
    /// Analyze stereo field
    pub check_stereo: bool,
    /// Analyze transients/pre-echo
    pub check_transients: bool,
    /// Analyze phase
    pub check_phase: bool,
    /// Run MFCC analysis
    pub check_mfcc: bool,
    /// Minimum confidence to report a defect
    pub min_confidence: f32,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        DetectionConfig {
            expected_bit_depth: 24,
            check_upsampling: true,
            check_stereo: true,
            check_transients: true,
            check_phase: false,  // Slower, disabled by default
            check_mfcc: false,   // Experimental, disabled by default
            min_confidence: 0.5,
        }
    }
}

/// Perform comprehensive quality analysis
pub fn detect_quality_issues(
    audio: &AudioData,
    config: &DetectionConfig,
) -> Result<QualityReport> {
    let mut defects = Vec::new();
    
    // ===== Spectral Analysis =====
    let mut spectral_analyzer = SpectralAnalyzer::new(8192, 2048, audio.sample_rate);
    let spectral_analysis = spectral_analyzer.analyze(audio)?;
    
    // Detect transcodes from spectral signature (CORRECTED v2)
    detect_transcode_from_spectral(&spectral_analysis, audio, &mut defects);
    
    // ===== Bit Depth Analysis =====
    let bit_depth_analysis = analyze_bit_depth(audio);
    
    if bit_depth_analysis.is_mismatch {
        defects.push(DetectedDefect {
            defect_type: DefectType::BitDepthMismatch {
                claimed: bit_depth_analysis.claimed_bit_depth,
                actual: bit_depth_analysis.actual_bit_depth,
                confidence: bit_depth_analysis.confidence,
            },
            confidence: bit_depth_analysis.confidence,
            evidence: bit_depth_analysis.evidence.clone(),
        });
    }
    
    // ===== Upsampling Analysis =====
    let upsampling_analysis = if config.check_upsampling {
        analyze_upsampling(audio, &spectral_analysis)
    } else {
        UpsamplingAnalysis {
            is_upsampled: false,
            original_sample_rate: None,
            current_sample_rate: audio.sample_rate,
            confidence: 0.0,
            detection_method: None,
            evidence: vec![],
            method_results: crate::upsampling::UpsamplingMethodResults {
                spectral_detected: false,
                spectral_original_rate: None,
                spectral_confidence: 0.0,
                null_test_detected: false,
                null_test_original_rate: None,
                null_test_confidence: 0.0,
                isp_detected: false,
                isp_confidence: 0.0,
            },
        }
    };
    
    if upsampling_analysis.is_upsampled && upsampling_analysis.confidence >= config.min_confidence {
        if let Some(original_rate) = upsampling_analysis.original_sample_rate {
            defects.push(DetectedDefect {
                defect_type: DefectType::Upsampled {
                    from: original_rate,
                    to: audio.sample_rate,
                    confidence: upsampling_analysis.confidence,
                },
                confidence: upsampling_analysis.confidence,
                evidence: upsampling_analysis.evidence.clone(),
            });
        }
    }
    
    // ===== Stereo Analysis =====
    let stereo_analysis = if config.check_stereo && audio.channels >= 2 {
        Some(analyze_stereo(audio))
    } else {
        None
    };
    
    if let Some(ref stereo) = stereo_analysis {
        if stereo.joint_stereo_detected && stereo.joint_stereo_confidence >= config.min_confidence {
            defects.push(DetectedDefect {
                defect_type: DefectType::JointStereo {
                    confidence: stereo.joint_stereo_confidence,
                },
                confidence: stereo.joint_stereo_confidence,
                evidence: stereo.evidence.clone(),
            });
        }
    }
    
    // ===== Pre-Echo Analysis =====
    let pre_echo_analysis = if config.check_transients {
        analyze_pre_echo(audio)
    } else {
        PreEchoAnalysis {
            transient_count: 0,
            pre_echo_count: 0,
            avg_pre_echo_level: -120.0,
            max_pre_echo_level: -120.0,
            pre_echo_score: 0.0,
            lossy_pre_echo_detected: false,
            confidence: 0.0,
            evidence: vec![],
        }
    };
    
    if pre_echo_analysis.lossy_pre_echo_detected && pre_echo_analysis.confidence >= config.min_confidence {
        defects.push(DetectedDefect {
            defect_type: DefectType::PreEcho {
                score: pre_echo_analysis.pre_echo_score,
            },
            confidence: pre_echo_analysis.confidence,
            evidence: pre_echo_analysis.evidence.clone(),
        });
    }
    
    // ===== Phase Analysis =====
    let phase_analysis = if config.check_phase {
        analyze_phase(audio)
    } else {
        PhaseAnalysis {
            phase_coherence: 1.0,
            discontinuity_score: 0.0,
            phase_jump_count: 0,
            codec_artifacts_likely: false,
            confidence: 0.0,
            evidence: vec![],
        }
    };
    
    if phase_analysis.codec_artifacts_likely && phase_analysis.confidence >= config.min_confidence {
        defects.push(DetectedDefect {
            defect_type: DefectType::PhaseDiscontinuities {
                score: phase_analysis.discontinuity_score,
            },
            confidence: phase_analysis.confidence,
            evidence: phase_analysis.evidence.clone(),
        });
    }
    
    // ===== True Peak Analysis =====
    let true_peak_analysis = analyze_true_peak(audio);
    
    if true_peak_analysis.has_clipping {
        defects.push(DetectedDefect {
            defect_type: DefectType::Clipping {
                percentage: true_peak_analysis.clipping_percentage,
            },
            confidence: 0.95,
            evidence: vec![format!("{:.3}% of samples clipped", 
                true_peak_analysis.clipping_percentage * 100.0)],
        });
    }
    
    if true_peak_analysis.has_inter_sample_overs {
        defects.push(DetectedDefect {
            defect_type: DefectType::InterSampleOvers {
                count: true_peak_analysis.inter_sample_over_count,
                max_level_db: true_peak_analysis.max_over_level,
            },
            confidence: 0.9,
            evidence: vec![format!("{} inter-sample overs, max {:.1} dB over",
                true_peak_analysis.inter_sample_over_count,
                true_peak_analysis.max_over_level)],
        });
    }
    
    // ===== Spectral Artifacts =====
    if spectral_analysis.has_artifacts {
        defects.push(DetectedDefect {
            defect_type: DefectType::SpectralArtifacts {
                artifact_score: spectral_analysis.artifact_score,
            },
            confidence: 0.7,
            evidence: vec![format!("Artifact score: {:.2}", spectral_analysis.artifact_score)],
        });
    }
    
    // ===== Calculate Overall Quality Score =====
    let quality_score = calculate_quality_score(&defects);
    let is_likely_lossless = quality_score > 0.7 && defects.iter()
        .filter(|d| matches!(d.defect_type, 
            DefectType::Mp3Transcode { .. } |
            DefectType::AacTranscode { .. } |
            DefectType::OggVorbisTranscode { .. } |
            DefectType::OpusTranscode { .. } |
            DefectType::BitDepthMismatch { .. } |
            DefectType::Upsampled { .. }
        ))
        .count() == 0;
    
    // Filter defects by confidence threshold
    let defects: Vec<DetectedDefect> = defects.into_iter()
        .filter(|d| d.confidence >= config.min_confidence)
        .collect();
    
    Ok(QualityReport {
        sample_rate: audio.sample_rate,
        channels: audio.channels,
        claimed_bit_depth: audio.claimed_bit_depth,
        actual_bit_depth: bit_depth_analysis.actual_bit_depth,
        duration_secs: audio.duration_secs,
        codec_name: audio.codec_name.clone(),
        
        frequency_cutoff: spectral_analysis.frequency_cutoff,
        spectral_rolloff: spectral_analysis.rolloff_95,
        rolloff_steepness: spectral_analysis.rolloff_steepness,
        has_brick_wall: spectral_analysis.has_brick_wall,
        spectral_flatness: spectral_analysis.spectral_flatness,
        
        dynamic_range: true_peak_analysis.loudness_info.dynamic_range_db,
        noise_floor: true_peak_analysis.loudness_info.rms_dbfs,
        peak_amplitude: true_peak_analysis.sample_peak_dbfs,
        true_peak: true_peak_analysis.true_peak_dbfs,
        crest_factor: true_peak_analysis.loudness_info.crest_factor_db,
        
        stereo_width: stereo_analysis.as_ref().map(|s| s.stereo_width),
        channel_correlation: stereo_analysis.as_ref().map(|s| s.channel_correlation),
        
        defects,
        quality_score,
        is_likely_lossless,
        
        spectral_analysis,
        bit_depth_analysis,
        stereo_analysis,
        pre_echo_analysis,
        phase_analysis,
        upsampling_analysis,
        true_peak_analysis,
    })
}

// =============================================================================
// CORRECTED v2: Transcode detection with proper high sample rate handling
// =============================================================================

/// Detect transcodes from spectral analysis - CORRECTED VERSION
/// 
/// KEY INSIGHT: For high sample rate files (88.2kHz+), we must use ABSOLUTE
/// frequency thresholds, not ratio-based detection. Music content naturally
/// stops around 20kHz (human hearing limit), so a 96kHz file with content
/// to 20kHz has only a 42% cutoff ratio - which would trigger false positives
/// with the old 85% threshold.
fn detect_transcode_from_spectral(
    spectral: &SpectralAnalysis,
    audio: &AudioData,
    defects: &mut Vec<DetectedDefect>,
) {
    let nyquist = audio.sample_rate as f32 / 2.0;
    let cutoff_hz = spectral.frequency_cutoff;
    let cutoff_ratio = cutoff_hz / nyquist;
    
    let is_high_sample_rate = audio.sample_rate >= 88200;
    
    // =========================================================================
    // HIGH SAMPLE RATE FILES (88.2kHz+): Use absolute frequency thresholds
    // =========================================================================
    if is_high_sample_rate {
        // Content up to 22kHz is NORMAL for any sample rate
        // (human hearing tops out at ~20kHz, instruments rarely exceed 22kHz)
        if cutoff_hz > 22000.0 {
            return;  // Normal high-res content
        }
        
        // 20-22kHz: need VERY strong evidence (brick-wall AND steep rolloff)
        if cutoff_hz >= 20000.0 {
            if !(spectral.has_brick_wall && spectral.rolloff_steepness > 80.0) {
                return;  // Likely natural rolloff at hearing limit
            }
        }
        
        // 18-20kHz: need strong evidence (brick-wall OR very steep rolloff)
        if cutoff_hz >= 18000.0 {
            if !spectral.has_brick_wall && spectral.rolloff_steepness < 60.0 {
                return;  // Could be mastering choice or natural content
            }
        }
        
        // 15-18kHz: require multiple signals
        if cutoff_hz >= 15000.0 {
            let evidence_count = 
                (if spectral.has_brick_wall { 1 } else { 0 }) +
                (if spectral.rolloff_steepness > 50.0 { 1 } else { 0 }) +
                (if spectral.has_shelf_pattern { 1 } else { 0 });
            
            if evidence_count < 2 {
                return;  // Insufficient evidence
            }
        }
        
        // 10-15kHz: look for codec-specific signatures, not just low cutoff
        // Renaissance choral, old jazz, ambient music can legitimately be here
        if cutoff_hz < 15000.0 && cutoff_hz >= 10000.0 {
            if !has_codec_signature(spectral, cutoff_hz) {
                return;  // Low content but no codec signature
            }
        }
        
        // Below 10kHz: still check for natural causes
        if cutoff_hz < 10000.0 {
            if !spectral.has_brick_wall {
                return;  // Not a sharp cutoff - might be natural
            }
        }
        
    } else {
        // =====================================================================
        // STANDARD SAMPLE RATES (44.1/48kHz): Use ratio-based thresholds
        // but more conservative than before
        // =====================================================================
        
        // At 44.1kHz, Nyquist is 22.05kHz
        // Content to 20kHz = 91% ratio - normal
        // Content to 18kHz = 82% ratio - could be MP3 320k or mastering
        // Content to 16kHz = 73% ratio - likely lossy transcode
        
        if cutoff_ratio >= 0.80 {
            return;  // Normal for standard sample rate
        }
        
        // 70-80% range: need evidence
        if cutoff_ratio >= 0.70 {
            if !spectral.has_brick_wall && spectral.rolloff_steepness < 40.0 {
                return;  // Probably natural rolloff
            }
        }
    }
    
    // =========================================================================
    // CONFIDENCE CALCULATION
    // =========================================================================
    
    let base_confidence: f32 = if is_high_sample_rate {
        // For high sample rate: confidence based on absolute cutoff + evidence
        if cutoff_hz < 12000.0 {
            0.90
        } else if cutoff_hz < 15000.0 {
            0.75
        } else if cutoff_hz < 18000.0 {
            0.60
        } else {
            0.50
        }
    } else {
        // For standard sample rate: ratio-based confidence
        if cutoff_ratio < 0.60 {
            0.90
        } else if cutoff_ratio < 0.70 {
            0.75
        } else {
            0.55
        }
    };
    
    // Evidence boost
    let evidence_boost: f32 = 
        (if spectral.has_brick_wall { 0.15 } else { 0.0 }) +
        (if spectral.rolloff_steepness > 60.0 { 0.10 } else { 0.0 }) +
        (if spectral.has_shelf_pattern { 0.10 } else { 0.0 });
    
    let confidence = (base_confidence + evidence_boost).min(0.95);
    
    // Confidence floor - raised from 0.4 to 0.55
    if confidence < 0.55 {
        return;
    }
    
    // =========================================================================
    // CODEC CLASSIFICATION (v2 - no default fallback)
    // =========================================================================
    
    let cutoff_hz_u32 = cutoff_hz as u32;
    
    let defect_type = match classify_codec_type_v2(spectral, cutoff_hz_u32) {
        Some(dt) => dt,
        None => return,  // Can't identify codec - DON'T FLAG
    };
    
    // Build evidence list
    let mut evidence = vec![
        format!("Frequency cutoff: {:.0} Hz", cutoff_hz),
    ];
    
    if is_high_sample_rate {
        evidence.push(format!(
            "High sample rate file ({}kHz) - content stops at {:.1}kHz", 
            audio.sample_rate / 1000,
            cutoff_hz / 1000.0
        ));
    } else {
        evidence.push(format!("Cutoff at {:.1}% of Nyquist", cutoff_ratio * 100.0));
    }
    
    evidence.push(format!("Rolloff steepness: {:.1} dB/octave", spectral.rolloff_steepness));
    
    if spectral.has_brick_wall {
        evidence.push("Brick-wall filter detected".to_string());
    }
    if spectral.has_shelf_pattern {
        evidence.push("Pre-cutoff shelf pattern (AAC characteristic)".to_string());
    }
    
    // Try to match against known signatures
    if let Some((name, sig_conf)) = match_signature(spectral) {
        evidence.push(format!("Matches {} signature ({:.0}%)", name, sig_conf * 100.0));
    }
    
    defects.push(DetectedDefect {
        defect_type,
        confidence,
        evidence,
    });
}

/// Check for codec-specific spectral signatures beyond just cutoff frequency
fn has_codec_signature(spectral: &SpectralAnalysis, cutoff_hz: f32) -> bool {
    // Brick-wall filter is strong indicator of lossy codec
    if spectral.has_brick_wall && spectral.rolloff_steepness > 40.0 {
        return true;
    }
    
    // AAC shelf pattern
    if spectral.has_shelf_pattern {
        return true;
    }
    
    // Opus has very specific bandwidth modes
    let opus_modes = [8000.0, 12000.0, 20000.0];
    for mode in opus_modes {
        if (cutoff_hz - mode).abs() < 500.0 && spectral.has_brick_wall {
            return true;
        }
    }
    
    // MP3 has specific bitrate-to-cutoff mapping
    let mp3_cutoffs = [
        (16000.0, 128),  // 128 kbps
        (18500.0, 192),  // 192 kbps
        (19500.0, 256),  // 256 kbps
        (20000.0, 320),  // 320 kbps
    ];
    for (freq, _) in mp3_cutoffs {
        if (cutoff_hz - freq).abs() < 500.0 && spectral.rolloff_steepness > 50.0 {
            return true;
        }
    }
    
    false
}

/// Classify codec type - version 2 with better discrimination and NO DEFAULT FALLBACK
fn classify_codec_type_v2(
    spectral: &SpectralAnalysis,
    cutoff_hz: u32,
) -> Option<DefectType> {
    
    // =========================================================================
    // MP3 Detection
    // Strong indicators: Brick-wall + very steep rolloff (>50 dB/oct)
    // Typical cutoffs: 15.5-20.5 kHz depending on bitrate
    // =========================================================================
    if spectral.has_brick_wall && spectral.rolloff_steepness > 50.0 {
        if cutoff_hz >= 15000 && cutoff_hz <= 20500 {
            let bitrate = estimate_mp3_bitrate(cutoff_hz);
            return Some(DefectType::Mp3Transcode { 
                cutoff_hz, 
                estimated_bitrate: bitrate,
            });
        }
    }
    
    // MP3 with slightly softer evidence (very steep rolloff alone)
    if spectral.rolloff_steepness > 70.0 && cutoff_hz >= 15000 && cutoff_hz <= 20500 {
        let bitrate = estimate_mp3_bitrate(cutoff_hz);
        return Some(DefectType::Mp3Transcode { 
            cutoff_hz, 
            estimated_bitrate: bitrate,
        });
    }
    
    // =========================================================================
    // AAC Detection
    // Strong indicator: Shelf pattern before cutoff
    // =========================================================================
    if spectral.has_shelf_pattern {
        let bitrate = estimate_aac_bitrate(cutoff_hz);
        return Some(DefectType::AacTranscode {
            cutoff_hz,
            estimated_bitrate: bitrate,
        });
    }
    
    // =========================================================================
    // Opus Detection
    // Very specific bandwidth modes with brick-wall
    // =========================================================================
    if spectral.has_brick_wall {
        // Wideband mode (8kHz)
        if cutoff_hz >= 7500 && cutoff_hz <= 8500 {
            return Some(DefectType::OpusTranscode { 
                cutoff_hz, 
                mode: "Wideband (8kHz)".to_string(),
            });
        }
        // Super-wideband mode (12kHz)
        if cutoff_hz >= 11500 && cutoff_hz <= 12500 {
            return Some(DefectType::OpusTranscode { 
                cutoff_hz, 
                mode: "Super-wideband (12kHz)".to_string(),
            });
        }
        // Fullband mode - harder to distinguish from MP3 320k
        if cutoff_hz >= 19500 && cutoff_hz <= 20500 {
            // Opus fullband typically has less steep rolloff than MP3
            if spectral.rolloff_steepness < 60.0 {
                return Some(DefectType::OpusTranscode { 
                    cutoff_hz, 
                    mode: "Fullband (20kHz)".to_string(),
                });
            }
        }
    }
    
    // =========================================================================
    // Vorbis Detection
    // ONLY flag if we have positive Vorbis-specific evidence
    // NO MORE DEFAULT FALLBACK!
    // =========================================================================
    // Vorbis characteristics:
    // - Softer rolloff than MP3 (typically 20-40 dB/oct)
    // - No sharp brick-wall
    // - Quality-dependent cutoff (Q3 ~14kHz, Q6 ~19kHz, Q10 ~22kHz)
    
    if !spectral.has_brick_wall 
        && spectral.rolloff_steepness >= 15.0 
        && spectral.rolloff_steepness <= 45.0 
        && cutoff_hz >= 12000 
        && cutoff_hz <= 19000 
    {
        // Estimate quality from cutoff
        let estimated_quality = if cutoff_hz < 14000 {
            Some(3)
        } else if cutoff_hz < 16000 {
            Some(5)
        } else if cutoff_hz < 18000 {
            Some(6)
        } else {
            Some(7)
        };
        
        // Only confident about lower quality settings
        if let Some(q) = estimated_quality {
            if q <= 6 {
                return Some(DefectType::OggVorbisTranscode {
                    cutoff_hz,
                    estimated_quality,
                });
            }
        }
    }
    
    // =========================================================================
    // NO DEFAULT FALLBACK
    // If we can't positively identify a codec, don't flag.
    // This prevents false positives on:
    // - Naturally band-limited content (ambient, classical, old recordings)
    // - Mastering choices that limit HF content
    // - High sample rate files with content only to 20kHz
    // =========================================================================
    None
}

/// Estimate MP3 bitrate from cutoff frequency
fn estimate_mp3_bitrate(cutoff_hz: u32) -> Option<u32> {
    match cutoff_hz {
        0..=11000 => Some(64),
        11001..=14000 => Some(96),
        14001..=16000 => Some(128),
        16001..=17500 => Some(160),
        17501..=18500 => Some(192),
        18501..=19500 => Some(224),
        19501..=20000 => Some(256),
        20001..=20500 => Some(320),
        _ => None,
    }
}

/// Estimate AAC bitrate from cutoff frequency
fn estimate_aac_bitrate(cutoff_hz: u32) -> Option<u32> {
    match cutoff_hz {
        0..=12000 => Some(64),
        12001..=15000 => Some(96),
        15001..=16500 => Some(128),
        16501..=18000 => Some(192),
        18001..=19500 => Some(256),
        19501..=21000 => Some(320),
        _ => None,
    }
}

/// Calculate overall quality score from defects
fn calculate_quality_score(defects: &[DetectedDefect]) -> f32 {
    if defects.is_empty() {
        return 1.0;
    }
    
    let mut score = 1.0f32;
    
    for defect in defects {
        let penalty = match &defect.defect_type {
            DefectType::Mp3Transcode { .. } => 0.8,
            DefectType::AacTranscode { .. } => 0.75,
            DefectType::OggVorbisTranscode { .. } => 0.7,
            DefectType::OpusTranscode { .. } => 0.7,
            DefectType::BitDepthMismatch { .. } => 0.5,
            DefectType::Upsampled { .. } => 0.4,
            DefectType::SpectralArtifacts { .. } => 0.2,
            DefectType::JointStereo { .. } => 0.15,
            DefectType::PreEcho { .. } => 0.3,
            DefectType::PhaseDiscontinuities { .. } => 0.2,
            DefectType::Clipping { .. } => 0.1,
            DefectType::InterSampleOvers { .. } => 0.05,
            DefectType::LowQuality { .. } => 0.3,
        };
        
        score *= 1.0 - (penalty * defect.confidence);
    }
    
    score.max(0.0)
}

/// Legacy function for backward compatibility
pub fn detect_quality_issues_simple(
    audio: &AudioData,
    expected_bit_depth: u32,
    check_upsampling: bool,
) -> Result<QualityReport> {
    let config = DetectionConfig {
        expected_bit_depth,
        check_upsampling,
        ..Default::default()
    };
    
    detect_quality_issues(audio, &config)
}
