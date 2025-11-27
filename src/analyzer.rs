// src/analyzer.rs
//
// High-level audio analyzer interface.
// Provides a simple API for analyzing audio files.

use anyhow::Result;
use std::path::Path;
use crate::decoder::{decode_audio, AudioData};
use crate::detector::{QualityReport, DetectionConfig, detect_quality_issues};
use crate::spectrogram::generate_spectrogram_image;

/// Audio analyzer with configurable options
pub struct AudioAnalyzer {
    audio_data: AudioData,
    config: DetectionConfig,
}

impl AudioAnalyzer {
    /// Create a new analyzer by loading an audio file
    pub fn new(path: &Path) -> Result<Self> {
        let audio_data = decode_audio(path)?;
        Ok(Self { 
            audio_data,
            config: DetectionConfig::default(),
        })
    }
    
    /// Create analyzer with custom configuration
    pub fn with_config(path: &Path, config: DetectionConfig) -> Result<Self> {
        let audio_data = decode_audio(path)?;
        Ok(Self { audio_data, config })
    }

    /// Get reference to audio data
    pub fn audio_data(&self) -> &AudioData {
        &self.audio_data
    }
    
    /// Set detection configuration
    pub fn set_config(&mut self, config: DetectionConfig) {
        self.config = config;
    }

    /// Perform full quality analysis
    pub fn analyze(&self) -> Result<QualityReport> {
        detect_quality_issues(&self.audio_data, &self.config)
    }
    
    /// Analyze with specific bit depth expectation
    pub fn analyze_with_bit_depth(&self, expected_bit_depth: u32, check_upsampling: bool) -> Result<QualityReport> {
        let mut config = self.config.clone();
        config.expected_bit_depth = expected_bit_depth;
        config.check_upsampling = check_upsampling;
        detect_quality_issues(&self.audio_data, &config)
    }

    /// Generate spectrogram image
    pub fn generate_spectrogram(
        &self, 
        output_path: &Path, 
        use_linear_scale: bool, 
        full_length: bool,
    ) -> Result<()> {
        generate_spectrogram_image(&self.audio_data, output_path, use_linear_scale, full_length)
    }
    
    /// Quick check - returns true if file appears to be lossless quality
    pub fn is_likely_lossless(&self) -> Result<bool> {
        let report = self.analyze()?;
        Ok(report.is_likely_lossless)
    }
    
    /// Get basic file information without full analysis
    pub fn get_file_info(&self) -> FileInfo {
        FileInfo {
            sample_rate: self.audio_data.sample_rate,
            channels: self.audio_data.channels,
            bit_depth: self.audio_data.claimed_bit_depth,
            duration_secs: self.audio_data.duration_secs,
            codec: self.audio_data.codec_name.clone(),
            total_samples: self.audio_data.samples.len(),
        }
    }
}

/// Basic file information
#[derive(Debug, Clone)]
pub struct FileInfo {
    pub sample_rate: u32,
    pub channels: usize,
    pub bit_depth: u32,
    pub duration_secs: f64,
    pub codec: String,
    pub total_samples: usize,
}

/// Builder pattern for analyzer configuration
pub struct AnalyzerBuilder {
    config: DetectionConfig,
}

impl AnalyzerBuilder {
    pub fn new() -> Self {
        Self {
            config: DetectionConfig::default(),
        }
    }
    
    pub fn expected_bit_depth(mut self, depth: u32) -> Self {
        self.config.expected_bit_depth = depth;
        self
    }
    
    pub fn check_upsampling(mut self, check: bool) -> Self {
        self.config.check_upsampling = check;
        self
    }
    
    pub fn check_stereo(mut self, check: bool) -> Self {
        self.config.check_stereo = check;
        self
    }
    
    pub fn check_transients(mut self, check: bool) -> Self {
        self.config.check_transients = check;
        self
    }
    
    pub fn check_phase(mut self, check: bool) -> Self {
        self.config.check_phase = check;
        self
    }
    
    pub fn check_mfcc(mut self, check: bool) -> Self {
        self.config.check_mfcc = check;
        self
    }
    
    pub fn min_confidence(mut self, confidence: f32) -> Self {
        self.config.min_confidence = confidence;
        self
    }
    
    pub fn build(self, path: &Path) -> Result<AudioAnalyzer> {
        AudioAnalyzer::with_config(path, self.config)
    }
}

impl Default for AnalyzerBuilder {
    fn default() -> Self {
        Self::new()
    }
}
