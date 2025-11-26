// tests/integration_test.rs
//
// Integration tests for the audio quality checker.
// These tests require the TestFiles directory with actual audio files.

use std::env;
use std::process::Command;
use std::path::{Path, PathBuf};

struct TestCase {
    file_path: String,
    should_pass: bool,
    expected_defects: Vec<String>,
    description: String,
}

struct TestResult {
    passed: bool,
    expected: bool,
    defects_found: Vec<String>,
    file: String,
}

#[test]
#[ignore]  // Only run when TestFiles directory is present
fn test_all_audio_files() {
    let binary_path = get_binary_path();
    
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let test_base = project_root.join("TestFiles");
    
    if !test_base.exists() {
        println!("TestFiles directory not found, skipping integration tests");
        return;
    }

    println!("\n=== Using TestFiles from: {} ===\n", test_base.display());

    let test_cases = define_test_cases(&test_base);
    let mut results = Vec::new();
    let mut passed = 0;
    let mut failed = 0;

    println!("=== Running Audio Quality Checker Integration Tests ===\n");

    for test_case in &test_cases {
        if !Path::new(&test_case.file_path).exists() {
            println!("⊘ SKIP: {} (file not found)", test_case.description);
            continue;
        }
        
        let result = run_test(&binary_path, test_case);
        
        if result.passed == result.expected {
            passed += 1;
            println!("✓ PASS: {}", test_case.description);
        } else {
            failed += 1;
            println!("✗ FAIL: {}", test_case.description);
            println!("  File: {}", test_case.file_path);
            println!("  Expected: {}, Got: {}", 
                if test_case.should_pass { "CLEAN" } else { "DEFECTS" },
                if result.passed { "CLEAN" } else { "DEFECTS" });
            if !result.defects_found.is_empty() {
                println!("  Defects found: {:?}", result.defects_found);
            }
            if !test_case.expected_defects.is_empty() {
                println!("  Expected defects: {:?}", test_case.expected_defects);
            }
        }
        
        results.push(result);
    }

    println!("\n=== Test Summary ===");
    println!("Total: {}", test_cases.len());
    println!("Passed: {}", passed);
    println!("Failed: {}", failed);
    if passed + failed > 0 {
        println!("Success Rate: {:.1}%", (passed as f32 / (passed + failed) as f32) * 100.0);
    }

    assert_eq!(failed, 0, "{} test(s) failed", failed);
}

fn get_binary_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");
    
    let release_path = path.join("release").join("audio-quality-checker");
    let debug_path = path.join("debug").join("audio-quality-checker");
    
    #[cfg(windows)]
    {
        let release_path = release_path.with_extension("exe");
        let debug_path = debug_path.with_extension("exe");
        
        if release_path.exists() {
            return release_path;
        } else if debug_path.exists() {
            return debug_path;
        }
    }
    
    #[cfg(not(windows))]
    {
        if release_path.exists() {
            return release_path;
        } else if debug_path.exists() {
            return debug_path;
        }
    }
    
    panic!("Binary not found. Please build the project first with: cargo build --release");
}

fn define_test_cases(base: &Path) -> Vec<TestCase> {
    let mut cases = Vec::new();

    // === CleanOrigin ===
    cases.push(TestCase {
        file_path: base.join("CleanOrigin/input96.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "Original 96kHz 24-bit FLAC (clean)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("CleanOrigin/input192.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "Original 192kHz file (actually 16-bit)".to_string(),
    });

    // === CleanTranscoded ===
    cases.push(TestCase {
        file_path: base.join("CleanTranscoded/input96_16bit.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "Honest 16-bit downsample from 96kHz".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("CleanTranscoded/input192_16bit.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "Honest 16-bit downsample from 192kHz".to_string(),
    });

    // === Resample96 - Downsamples should pass ===
    for rate in &["44", "48", "88"] {
        cases.push(TestCase {
            file_path: base.join(format!("Resample96/input96_{}.flac", rate)).to_string_lossy().to_string(),
            should_pass: true,
            expected_defects: vec![],
            description: format!("Downsampled 96kHz to {}kHz (clean)", rate),
        });
    }

    // === Resample192 - All from 16-bit source ===
    for rate in &["44", "48", "88", "96", "176"] {
        cases.push(TestCase {
            file_path: base.join(format!("Resample192/input192_{}.flac", rate)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec!["BitDepthMismatch".to_string()],
            description: format!("Resampled 192kHz to {}kHz (16-bit source)", rate),
        });
    }

    // === Upscale16 ===
    cases.push(TestCase {
        file_path: base.join("Upscale16/output96_16bit.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "16-bit upscaled to 24-bit".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscale16/output192_16bit.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "16-bit upscaled to 24-bit from 192kHz".to_string(),
    });

    // === Upscaled - Lossy transcodes ===
    let lossy_formats = vec![
        ("mp3", "Mp3"),
        ("m4a", "Aac"),
        ("opus", "Opus"),
        ("ogg", "Vorbis"),
    ];

    for (format, defect) in &lossy_formats {
        cases.push(TestCase {
            file_path: base.join(format!("Upscaled/input96_{}.flac", format)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec![defect.to_string()],
            description: format!("{} from 96kHz transcoded to FLAC", format.to_uppercase()),
        });

        cases.push(TestCase {
            file_path: base.join(format!("Upscaled/input192_{}.flac", format)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec![defect.to_string()],
            description: format!("{} from 192kHz transcoded to FLAC", format.to_uppercase()),
        });
    }

    // === MasterScript ===
    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_original.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "MasterScript: Original 96kHz reference (clean)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("MasterScript/test192_original.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "MasterScript: Original 192kHz (16-bit source)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_16bit_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "MasterScript: 16-bit upscaled to 24-bit".to_string(),
    });

    cases
}

fn run_test(binary: &Path, test_case: &TestCase) -> TestResult {
    let output = Command::new(binary)
        .arg("--input")
        .arg(&test_case.file_path)
        .arg("--bit-depth")
        .arg("24")
        .arg("--check-upsampling")
        .arg("--verbose")
        .output()
        .expect("Failed to execute binary");

    let stdout = String::from_utf8_lossy(&output.stdout);

    let has_issues = stdout.contains("✗ ISSUES DETECTED") || 
                     stdout.contains("ISSUES DETECTED");
    
    let is_clean = stdout.contains("✓ CLEAN") || 
                   (stdout.contains("CLEAN") && !has_issues);

    let mut defects_found = Vec::new();
    
    if stdout.contains("MP3") || stdout.contains("Mp3") {
        defects_found.push("Mp3".to_string());
    }
    if stdout.contains("AAC") || stdout.contains("Aac") {
        defects_found.push("Aac".to_string());
    }
    if stdout.contains("Opus") {
        defects_found.push("Opus".to_string());
    }
    if stdout.contains("Vorbis") || stdout.contains("Ogg") {
        defects_found.push("Vorbis".to_string());
    }
    if stdout.contains("Bit depth mismatch") || stdout.contains("BitDepth") {
        defects_found.push("BitDepthMismatch".to_string());
    }
    if stdout.contains("Upsampled") {
        defects_found.push("Upsampled".to_string());
    }
    if stdout.contains("Spectral artifacts") {
        defects_found.push("SpectralArtifacts".to_string());
    }

    TestResult {
        passed: is_clean,
        expected: test_case.should_pass,
        defects_found,
        file: test_case.file_path.clone(),
    }
}

#[test]
fn test_binary_exists() {
    let binary_path = get_binary_path();
    assert!(binary_path.exists(), "Binary not found at {:?}", binary_path);
}

#[test]
fn test_help_output() {
    let binary_path = get_binary_path();
    let output = Command::new(&binary_path)
        .arg("--help")
        .output()
        .expect("Failed to execute binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("audio-quality-checker"));
    assert!(stdout.contains("--input"));
    assert!(stdout.contains("--bit-depth"));
}

// Unit tests for the library
#[cfg(test)]
mod unit_tests {
    use audio_quality_checker::dsp::*;
    
    #[test]
    fn test_window_functions() {
        let hann = create_window(1024, WindowType::Hann);
        assert_eq!(hann.len(), 1024);
        assert!(hann[0] < 0.01);  // Edges should be near zero
        assert!(hann[512] > 0.99);  // Center should be near 1
        
        let blackman = create_window(1024, WindowType::Blackman);
        assert_eq!(blackman.len(), 1024);
    }
    
    #[test]
    fn test_rms() {
        let samples = vec![1.0, -1.0, 1.0, -1.0];
        let result = rms(&samples);
        assert!((result - 1.0).abs() < 0.001);
        
        let zeros = vec![0.0; 100];
        assert_eq!(rms(&zeros), 0.0);
    }
    
    #[test]
    fn test_amplitude_conversion() {
        assert!((amplitude_to_db(1.0) - 0.0).abs() < 0.001);
        assert!((amplitude_to_db(0.5) - (-6.02)).abs() < 0.1);
        
        assert!((db_to_amplitude(0.0) - 1.0).abs() < 0.001);
        assert!((db_to_amplitude(-6.0) - 0.5).abs() < 0.01);
    }
}
