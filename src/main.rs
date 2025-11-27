// src/main.rs
//
// Audio Quality Checker - Detect fake lossless, transcodes, and upsampled audio.
// Uses advanced DSP techniques for comprehensive audio analysis.

use anyhow::Result;
use clap::Parser;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use colorful::Colorful;

mod analyzer;
mod decoder;
mod dsp;
mod spectrogram;
mod spectral;
mod bit_depth;
mod stereo;
mod transients;
mod phase;
mod upsampling;
mod true_peak;
mod mfcc;
mod detector;

use analyzer::AudioAnalyzer;
use detector::{QualityReport, DetectedDefect, DefectType, DetectionConfig};

#[derive(Parser, Debug)]
#[command(name = "audio-quality-checker")]
#[command(version = "0.2.0")]
#[command(about = "Detect fake lossless, transcodes, and upsampled audio files with advanced spectral analysis")]
struct Args {
    /// Input file or directory (defaults to current directory)
    #[arg(short, long, default_value = ".")]
    input: PathBuf,

    /// Expected bit depth (16 or 24)
    #[arg(short, long, default_value = "24")]
    bit_depth: u32,

    /// Generate spectrogram images
    #[arg(short, long)]
    spectrogram: bool,

    /// Use linear frequency scale instead of mel scale
    #[arg(long)]
    linear_scale: bool,

    /// Generate full-length spectrogram instead of first 15 seconds
    #[arg(long)]
    full_spectrogram: bool,

    /// Output mode: "source" (same as audio file), "current" (cwd), or custom path
    #[arg(short, long, default_value = "source")]
    output: String,

    /// Check for upsampling
    #[arg(short = 'u', long)]
    check_upsampling: bool,

    /// Enable stereo analysis
    #[arg(long)]
    stereo: bool,

    /// Enable transient/pre-echo analysis
    #[arg(long)]
    transients: bool,

    /// Enable phase analysis (slower)
    #[arg(long)]
    phase: bool,

    /// Verbose output with detailed analysis
    #[arg(short, long)]
    verbose: bool,

    /// Output results as JSON
    #[arg(long)]
    json: bool,

    /// Minimum confidence threshold (0.0 - 1.0)
    #[arg(long, default_value = "0.5")]
    min_confidence: f32,

    /// Quick mode - skip slower analyses
    #[arg(short, long)]
    quick: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.spectrogram && args.output != "source" && args.output != "current" {
        std::fs::create_dir_all(&args.output)?;
    }

    let audio_files = collect_audio_files(&args.input)?;

    if audio_files.is_empty() {
        if !args.json {
            println!("{}", "No audio files found!".red());
        }
        return Ok(());
    }

    if !args.json {
        println!("Found {} audio file(s)\n", audio_files.len());
    }

    let mut all_results = Vec::new();

    for file_path in audio_files {
        match process_file(&file_path, &args) {
            Ok(result) => all_results.push(result),
            Err(e) => {
                if !args.json {
                    eprintln!("{}: {}", "Error processing".red(), e);
                }
            }
        }
    }

    if args.json {
        // Output JSON summary
        let json_output = serde_json::json!({
            "files_analyzed": all_results.len(),
            "results": all_results.iter().map(|(path, report)| {
                serde_json::json!({
                    "file": path,
                    "sample_rate": report.sample_rate,
                    "channels": report.channels,
                    "claimed_bit_depth": report.claimed_bit_depth,
                    "actual_bit_depth": report.actual_bit_depth,
                    "duration_secs": report.duration_secs,
                    "frequency_cutoff": report.frequency_cutoff,
                    "quality_score": report.quality_score,
                    "is_likely_lossless": report.is_likely_lossless,
                    "defects": report.defects.iter().map(|d| {
                        serde_json::json!({
                            "type": format_defect_type(&d.defect_type),
                            "confidence": d.confidence,
                            "evidence": d.evidence,
                        })
                    }).collect::<Vec<_>>(),
                })
            }).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&json_output)?);
    }

    Ok(())
}

fn collect_audio_files(path: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let audio_extensions = ["flac", "wav", "mp3", "ogg", "m4a", "aac", "opus", "wv", "ape", "aiff", "aif"];

    if path.is_file() {
        if let Some(ext) = path.extension() {
            if audio_extensions.contains(&ext.to_str().unwrap_or("").to_lowercase().as_str()) {
                files.push(path.to_path_buf());
            }
        }
    } else if path.is_dir() {
        for entry in WalkDir::new(path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let entry_path = entry.path();
            if let Some(ext) = entry_path.extension() {
                if audio_extensions.contains(&ext.to_str().unwrap_or("").to_lowercase().as_str()) {
                    files.push(entry_path.to_path_buf());
                }
            }
        }
    }

    Ok(files)
}

fn process_file(file_path: &Path, args: &Args) -> Result<(String, QualityReport)> {
    if !args.json {
        println!("Analyzing: {}", file_path.display().to_string().cyan());
    }

    // Build detection configuration
    let config = DetectionConfig {
        expected_bit_depth: args.bit_depth,
        check_upsampling: args.check_upsampling,
        check_stereo: args.stereo || !args.quick,
        check_transients: args.transients || !args.quick,
        check_phase: args.phase,
        check_mfcc: false,
        min_confidence: args.min_confidence,
    };

    let analyzer = AudioAnalyzer::with_config(file_path, config)?;
    let report = analyzer.analyze()?;

    if !args.json {
        print_report(&report, args.verbose);
    }

    if args.spectrogram {
        let output_path = determine_output_path(file_path, &args.output)?;
        analyzer.generate_spectrogram(&output_path, args.linear_scale, args.full_spectrogram)?;
        if !args.json {
            println!("  Spectrogram: {}", output_path.display());
        }
    }

    if !args.json {
        println!();
    }

    Ok((file_path.display().to_string(), report))
}

fn determine_output_path(file_path: &Path, output_mode: &str) -> Result<PathBuf> {
    let stem = file_path.file_stem()
        .ok_or_else(|| anyhow::anyhow!("Invalid file name"))?
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid UTF-8 in file name"))?;

    let output_path = match output_mode {
        "source" => {
            let parent = file_path.parent().unwrap_or(Path::new("."));
            parent.join(format!("{}_spectrogram.png", stem))
        },
        "current" => {
            std::env::current_dir()?.join(format!("{}_spectrogram.png", stem))
        },
        custom_path => {
            let out_dir = PathBuf::from(custom_path);
            std::fs::create_dir_all(&out_dir)?;
            out_dir.join(format!("{}_spectrogram.png", stem))
        }
    };

    Ok(output_path)
}

fn print_report(report: &QualityReport, verbose: bool) {
    // Basic info
    println!("  Sample Rate: {} Hz", report.sample_rate);
    println!("  Bit Depth: {} bit (claimed: {})", 
        report.actual_bit_depth, report.claimed_bit_depth);
    println!("  Channels: {}", report.channels);
    println!("  Duration: {:.2}s", report.duration_secs);
    println!("  Frequency Cutoff: {:.0} Hz", report.frequency_cutoff);
    println!("  Dynamic Range: {:.1} dB", report.dynamic_range);

    // Quality score
    let score_str = format!("{:.0}%", report.quality_score * 100.0);
    let quality_label = if report.is_likely_lossless {
        format!("Quality Score: {} (likely lossless)", score_str).green()
    } else {
        format!("Quality Score: {} (likely transcoded)", score_str).yellow()
    };
    println!("  {}", quality_label);

    // Defects
    if report.defects.is_empty() {
        println!("  Status: {}", "✓ CLEAN - No issues detected".green());
    } else {
        println!("  Status: {}", "✗ ISSUES DETECTED".red());
        for defect in &report.defects {
            let defect_str = format_defect(defect);
            println!("    • {}", defect_str.yellow());
            
            if verbose {
                for evidence in &defect.evidence {
                    println!("      - {}", evidence.dark_gray());
                }
            }
        }
    }

    // Verbose details
    if verbose {
        println!("\n  {} Technical Details:", "▸".cyan());
        println!("    Spectral rolloff: {:.0} Hz", report.spectral_rolloff);
        println!("    Rolloff steepness: {:.1} dB/octave", report.rolloff_steepness);
        println!("    Brick-wall cutoff: {}", if report.has_brick_wall { "Yes" } else { "No" });
        println!("    Spectral flatness: {:.3}", report.spectral_flatness);
        println!("    Peak amplitude: {:.1} dBFS", report.peak_amplitude);
        println!("    True peak: {:.1} dBFS", report.true_peak);
        println!("    Crest factor: {:.1} dB", report.crest_factor);
        
        if let Some(width) = report.stereo_width {
            println!("    Stereo width: {:.1}%", width * 100.0);
        }
        if let Some(corr) = report.channel_correlation {
            println!("    Channel correlation: {:.3}", corr);
        }
        
        // Bit depth analysis details
        println!("\n  {} Bit Depth Analysis:", "▸".cyan());
        println!("    LSB method: {} bit", report.bit_depth_analysis.method_results.lsb_method);
        println!("    Histogram method: {} bit", report.bit_depth_analysis.method_results.histogram_method);
        println!("    Noise floor method: {} bit", report.bit_depth_analysis.method_results.noise_method);
        println!("    Clustering method: {} bit", report.bit_depth_analysis.method_results.clustering_method);
        println!("    Overall confidence: {:.1}%", report.bit_depth_analysis.confidence * 100.0);
        
        // Pre-echo analysis
        if report.pre_echo_analysis.transient_count > 0 {
            println!("\n  {} Pre-Echo Analysis:", "▸".cyan());
            println!("    Transients analyzed: {}", report.pre_echo_analysis.transient_count);
            println!("    Pre-echo detected: {}", report.pre_echo_analysis.pre_echo_count);
            println!("    Pre-echo score: {:.2}", report.pre_echo_analysis.pre_echo_score);
        }
        
        // Upsampling analysis
        if report.upsampling_analysis.is_upsampled {
            println!("\n  {} Upsampling Analysis:", "▸".cyan());
            if let Some(orig) = report.upsampling_analysis.original_sample_rate {
                println!("    Original rate: {} Hz", orig);
            }
            println!("    Detection method: {:?}", report.upsampling_analysis.detection_method);
            println!("    Confidence: {:.1}%", report.upsampling_analysis.confidence * 100.0);
        }
    }
}

fn format_defect(defect: &DetectedDefect) -> String {
    let conf_str = format!(" ({:.0}% confidence)", defect.confidence * 100.0);
    
    match &defect.defect_type {
        DefectType::Mp3Transcode { cutoff_hz, estimated_bitrate } => {
            let bitrate_str = estimated_bitrate
                .map(|b| format!(" (~{}kbps)", b))
                .unwrap_or_default();
            format!("MP3 transcode{} - cutoff at {} Hz{}", bitrate_str, cutoff_hz, conf_str)
        },
        DefectType::OggVorbisTranscode { cutoff_hz, estimated_quality } => {
            let quality_str = estimated_quality
                .map(|q| format!(" (~q{})", q))
                .unwrap_or_default();
            format!("Ogg Vorbis transcode{} - cutoff at {} Hz{}", quality_str, cutoff_hz, conf_str)
        },
        DefectType::AacTranscode { cutoff_hz, estimated_bitrate } => {
            let bitrate_str = estimated_bitrate
                .map(|b| format!(" (~{}kbps)", b))
                .unwrap_or_default();
            format!("AAC transcode{} - cutoff at {} Hz{}", bitrate_str, cutoff_hz, conf_str)
        },
        DefectType::OpusTranscode { cutoff_hz, mode } => {
            format!("Opus transcode ({}) - cutoff at {} Hz{}", mode, cutoff_hz, conf_str)
        },
        DefectType::BitDepthMismatch { claimed, actual, .. } => {
            format!("Bit depth mismatch: claims {}-bit, actually {}-bit{}", claimed, actual, conf_str)
        },
        DefectType::Upsampled { from, to, .. } => {
            format!("Upsampled from {} Hz to {} Hz{}", from, to, conf_str)
        },
        DefectType::SpectralArtifacts { artifact_score } => {
            format!("Spectral artifacts detected (score: {:.2}){}", artifact_score, conf_str)
        },
        DefectType::JointStereo { .. } => {
            format!("Joint stereo encoding detected{}", conf_str)
        },
        DefectType::PreEcho { score } => {
            format!("Pre-echo artifacts (score: {:.2}){}", score, conf_str)
        },
        DefectType::PhaseDiscontinuities { score } => {
            format!("Phase discontinuities (score: {:.2}){}", score, conf_str)
        },
        DefectType::Clipping { percentage } => {
            format!("Clipping detected ({:.3}% of samples){}", percentage * 100.0, conf_str)
        },
        DefectType::InterSampleOvers { count, max_level_db } => {
            format!("{} inter-sample overs (max: {:.1} dB){}", count, max_level_db, conf_str)
        },
        DefectType::LowQuality { description } => {
            format!("Low quality: {}{}", description, conf_str)
        },
    }
}

fn format_defect_type(defect: &DefectType) -> String {
    match defect {
        DefectType::Mp3Transcode { .. } => "MP3 Transcode".to_string(),
        DefectType::OggVorbisTranscode { .. } => "Ogg Vorbis Transcode".to_string(),
        DefectType::AacTranscode { .. } => "AAC Transcode".to_string(),
        DefectType::OpusTranscode { .. } => "Opus Transcode".to_string(),
        DefectType::BitDepthMismatch { .. } => "Bit Depth Mismatch".to_string(),
        DefectType::Upsampled { .. } => "Upsampled".to_string(),
        DefectType::SpectralArtifacts { .. } => "Spectral Artifacts".to_string(),
        DefectType::JointStereo { .. } => "Joint Stereo".to_string(),
        DefectType::PreEcho { .. } => "Pre-Echo".to_string(),
        DefectType::PhaseDiscontinuities { .. } => "Phase Discontinuities".to_string(),
        DefectType::Clipping { .. } => "Clipping".to_string(),
        DefectType::InterSampleOvers { .. } => "Inter-Sample Overs".to_string(),
        DefectType::LowQuality { .. } => "Low Quality".to_string(),
    }
}
