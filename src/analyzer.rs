// src/analyzer.rs

use anyhow::Result;
use std::path::Path;
use crate::decoder::{decode_audio, AudioData};
use crate::detector::{QualityReport, detect_quality_issues};
use crate::spectrogram::generate_spectrogram_image;

pub struct AudioAnalyzer {
    audio_data: AudioData,
}

impl AudioAnalyzer {
    pub fn new(path: &Path) -> Result<Self> {
        let audio_data = decode_audio(path)?;
        Ok(Self { audio_data })
    }

    pub fn analyze(&self, expected_bit_depth: u32, check_upsampling: bool) -> Result<QualityReport> {
        detect_quality_issues(&self.audio_data, expected_bit_depth, check_upsampling)
    }

    pub fn generate_spectrogram(&self, output_path: &Path, use_linear_scale: bool, full_length: bool) -> Result<()> {
        generate_spectrogram_image(&self.audio_data, output_path, use_linear_scale, full_length)
    }
}

