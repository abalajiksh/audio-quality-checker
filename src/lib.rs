// src/lib.rs
//
// Audio Quality Checker Library
// Detect fake lossless, transcodes, and upsampled audio using advanced DSP.

pub mod analyzer;
pub mod decoder;
pub mod dsp;
pub mod spectrogram;
pub mod spectral;
pub mod bit_depth;
pub mod stereo;
pub mod transients;
pub mod phase;
pub mod upsampling;
pub mod true_peak;
pub mod mfcc;
pub mod detector;

// Re-export main types for convenience
pub use analyzer::{AudioAnalyzer, AnalyzerBuilder, FileInfo};
pub use decoder::AudioData;
pub use detector::{
    QualityReport, DetectedDefect, DefectType, DetectionConfig,
    detect_quality_issues,
};
pub use spectral::SpectralAnalysis;
pub use bit_depth::BitDepthAnalysis;
pub use stereo::StereoAnalysis;
pub use transients::{PreEchoAnalysis, FrameBoundaryAnalysis};
pub use phase::PhaseAnalysis;
pub use upsampling::UpsamplingAnalysis;
pub use true_peak::TruePeakAnalysis;
pub use mfcc::MfccAnalysis;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Quick analysis - check if a file is likely lossless
/// 
/// # Example
/// ```ignore
/// use audio_quality_checker::is_likely_lossless;
/// use std::path::Path;
/// 
/// let result = is_likely_lossless(Path::new("audio.flac"));
/// println!("Likely lossless: {:?}", result);
/// ```
pub fn is_likely_lossless(path: &std::path::Path) -> anyhow::Result<bool> {
    let analyzer = AudioAnalyzer::new(path)?;
    analyzer.is_likely_lossless()
}

/// Analyze audio file with default settings
/// 
/// # Example
/// ```ignore
/// use audio_quality_checker::analyze_file;
/// use std::path::Path;
/// 
/// let report = analyze_file(Path::new("audio.flac"))?;
/// println!("Quality score: {}", report.quality_score);
/// ```
pub fn analyze_file(path: &std::path::Path) -> anyhow::Result<QualityReport> {
    let analyzer = AudioAnalyzer::new(path)?;
    analyzer.analyze()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
