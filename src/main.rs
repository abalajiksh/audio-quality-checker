// src/main.rs

use anyhow::Result;
use clap::Parser;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use colorful::Colorful;

mod analyzer;
mod decoder;
mod spectrogram;
mod detector;

use analyzer::AudioAnalyzer;
use detector::{QualityReport, DefectType};

#[derive(Parser, Debug)]
#[command(name = "audio-quality-checker")]
#[command(about = "Detect fake lossless, transcodes, and upsampled audio files")]
struct Args {
    /// Input file or directory (defaults to current directory if not specified)
    #[arg(short, long, default_value = ".")]
    input: PathBuf,

    /// Expected bit depth (16 or 24)
    #[arg(short, long, default_value = "24")]
    bit_depth: u32,

    /// Generate spectrogram images
    #[arg(short, long)]
    spectrogram: bool,

    /// Use linear frequency scale instead of mel scale (produces longer spectrograms)
    #[arg(long)]
    linear_scale: bool,

    /// Generate full-length spectrogram instead of first 15 seconds
    #[arg(long)]
    full_spectrogram: bool,

    /// Output mode: "source" (same as audio file), "current" (current directory), or custom path
    #[arg(short, long, default_value = "source")]
    output: String,

    /// Check for upsampling
    #[arg(short = 'u', long)]
    check_upsampling: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Only create output dir if not using "source" mode
    if args.spectrogram && args.output != "source" {
        std::fs::create_dir_all(&args.output)?;
    }

    let audio_files = collect_audio_files(&args.input)?;

    if audio_files.is_empty() {
        println!("{}", "No audio files found!".red());
        return Ok(());
    }

    println!("Found {} audio file(s)\n", audio_files.len());

    for file_path in audio_files {
        process_file(&file_path, &args)?;
    }

    Ok(())
}

fn collect_audio_files(path: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let audio_extensions = ["flac", "wav", "mp3", "ogg", "m4a", "aac", "opus"];

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
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if audio_extensions.contains(&ext.to_str().unwrap_or("").to_lowercase().as_str()) {
                    files.push(path.to_path_buf());
                }
            }
        }
    }

    Ok(files)
}

fn process_file(file_path: &Path, args: &Args) -> Result<()> {
    println!("Analyzing: {}", file_path.display().to_string().cyan());

    let analyzer = AudioAnalyzer::new(file_path)?;
    let report = analyzer.analyze(args.bit_depth, args.check_upsampling)?;

    print_report(&report, args.verbose);

    if args.spectrogram {
        let output_path = match args.output.as_str() {
            "source" => {
                // Save next to the audio file
                let parent = file_path.parent().unwrap_or(Path::new("."));
                parent.join(format!(
                    "{}_spectrogram.png",
                    file_path.file_stem().unwrap().to_str().unwrap()
                ))
            },
            "current" => {
                // Save in current working directory
                std::env::current_dir()?.join(format!(
                    "{}_spectrogram.png",
                    file_path.file_stem().unwrap().to_str().unwrap()
                ))
            },
            custom_path => {
                // Save in specified directory
                let out_dir = PathBuf::from(custom_path);
                std::fs::create_dir_all(&out_dir)?;
                out_dir.join(format!(
                    "{}.png",
                    file_path.file_stem().unwrap().to_str().unwrap()
                ))
            }
        };

        analyzer.generate_spectrogram(&output_path, args.linear_scale, args.full_spectrogram)?;
        println!("  Spectrogram saved to: {}", output_path.display());
    }

    println!();
    Ok(())
}

fn print_report(report: &QualityReport, verbose: bool) {
    println!("  Sample Rate: {} Hz", report.sample_rate);
    println!("  Bit Depth: {} bit (claimed: {})", report.actual_bit_depth, report.claimed_bit_depth);
    println!("  Channels: {}", report.channels);
    println!("  Duration: {:.2}s", report.duration_secs);
    println!("  Frequency Cutoff: {:.0} Hz", report.frequency_cutoff);
    println!("  Dynamic Range: {:.1} dB", report.dynamic_range);

    if report.defects.is_empty() {
        println!("  Status: {}", "✓ CLEAN".green());
    } else {
        println!("  Status: {}", "✗ ISSUES DETECTED".red());
        for defect in &report.defects {
            println!("    • {}", format_defect(defect).yellow());
        }
    }

    if verbose {
        println!("\n  Technical Details:");
        println!("    Noise Floor: {:.1} dB", report.noise_floor);
        println!("    Peak Amplitude: {:.1} dB", report.peak_amplitude);
        println!("    Spectral Rolloff: {:.0} Hz", report.spectral_rolloff);
    }
}

fn format_defect(defect: &DefectType) -> String {
    match defect {
        DefectType::Mp3Transcode { cutoff_hz } =>
            format!("Likely MP3 transcode (cutoff at {} Hz)", cutoff_hz),
        DefectType::OggVorbisTranscode { cutoff_hz } =>
            format!("Likely Ogg Vorbis transcode (cutoff at {} Hz)", cutoff_hz),
        DefectType::AacTranscode { cutoff_hz } =>
            format!("Possible AAC/M4A transcode (cutoff at {} Hz)", cutoff_hz),
        DefectType::OpusTranscode { cutoff_hz, mode } =>
            format!("Likely Opus transcode - {} mode (cutoff at {} Hz)", mode, cutoff_hz),
        DefectType::BitDepthMismatch { claimed, actual } =>
            format!("Bit depth mismatch: claimed {}-bit, actually {}-bit", claimed, actual),
        DefectType::Upsampled { from, to } =>
            format!("Upsampled from {} Hz to {} Hz", from, to),
        DefectType::SpectralArtifacts =>
            "Spectral artifacts detected".to_string(),
        DefectType::LowQuality =>
            "Poor quality encoding detected".to_string(),
    }
}

