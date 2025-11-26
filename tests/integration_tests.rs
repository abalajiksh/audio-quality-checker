// tests/integration_test.rs

use std::process::Command;
use std::path::{Path, PathBuf};
use std::collections::HashMap;

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
fn test_all_audio_files() {
    let binary_path = get_binary_path();
    let test_base = PathBuf::from("../TestFiles");
    
    if !test_base.exists() {
        panic!("TestFiles directory not found. Expected at: {:?}", test_base.canonicalize());
    }

    let test_cases = define_test_cases(&test_base);
    let mut results = Vec::new();
    let mut passed = 0;
    let mut failed = 0;

    println!("\n=== Running Audio Quality Checker Integration Tests ===\n");

    for test_case in &test_cases {
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
        }
        
        results.push(result);
    }

    // Print summary
    println!("\n=== Test Summary ===");
    println!("Total: {}", test_cases.len());
    println!("Passed: {}", passed);
    println!("Failed: {}", failed);
    println!("Success Rate: {:.1}%", (passed as f32 / test_cases.len() as f32) * 100.0);

    // Print detailed failure analysis
    if failed > 0 {
        println!("\n=== Failure Analysis ===");
        for (i, result) in results.iter().enumerate() {
            if result.passed != result.expected {
                println!("\n{}: {}", i + 1, test_cases[i].description);
                println!("  File: {}", result.file);
                println!("  Expected defects: {:?}", test_cases[i].expected_defects);
                println!("  Found defects: {:?}", result.defects_found);
            }
        }
    }

    assert_eq!(failed, 0, "{} test(s) failed", failed);
}

fn get_binary_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");
    
    // Try release build first, then debug
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

    // CleanOrigin - Should PASS (pristine originals)
    cases.push(TestCase {
        file_path: base.join("CleanOrigin/input96.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "Original 96kHz 24-bit FLAC (should be clean)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("CleanOrigin/input192.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "Original 192kHz 24-bit FLAC (should be clean)".to_string(),
    });

    // CleanTranscoded - Should PASS (honest 16-bit)
    cases.push(TestCase {
        file_path: base.join("CleanTranscoded/input96_16bit.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "Honest 16-bit downsample from 96kHz (should pass)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("CleanTranscoded/input192_16bit.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "Honest 16-bit downsample from 192kHz (should pass)".to_string(),
    });

    // Resample96 - MIXED (downsamples pass, upsamples fail)
    let downsample_rates = vec!["44", "48", "88"];
    for rate in downsample_rates {
        cases.push(TestCase {
            file_path: base.join(format!("Resample96/input96_{}.flac", rate)).to_string_lossy().to_string(),
            should_pass: true,
            expected_defects: vec![],
            description: format!("Downsampled 96kHz to {}kHz (honest cutoff, should pass)", rate),
        });
    }

    let upsample_rates = vec!["176", "192"];
    for rate in upsample_rates {
        cases.push(TestCase {
            file_path: base.join(format!("Resample96/input96_{}.flac", rate)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec!["Upsampled".to_string()],
            description: format!("Upsampled 96kHz to {}kHz (fake hi-res, should fail)", rate),
        });
    }

    // Resample192 - ALL PASS (all are downsamples)
    let resample192_rates = vec!["44", "48", "88", "96", "176"];
    for rate in resample192_rates {
        cases.push(TestCase {
            file_path: base.join(format!("Resample192/input192_{}.flac", rate)).to_string_lossy().to_string(),
            should_pass: true,
            expected_defects: vec![],
            description: format!("Downsampled 192kHz to {}kHz (honest cutoff, should pass)", rate),
        });
    }

    // Upscale16 - ALL FAIL (16-bit upscaled to 24-bit)
    cases.push(TestCase {
        file_path: base.join("Upscale16/output96_16bit.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "16-bit upscaled to 24-bit (fake bit depth, should fail)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscale16/output192_16bit.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "16-bit upscaled to 24-bit from 192kHz (fake bit depth, should fail)".to_string(),
    });

    // Upscaled - ALL FAIL (lossy to FLAC)
    let lossy_formats = vec![
        ("mp3", "Mp3Transcode"),
        ("m4a", "AacTranscode"),
        ("opus", "OpusTranscode"),
        ("ogg", "OggVorbisTranscode"),
    ];

    for (format, defect) in &lossy_formats {
        cases.push(TestCase {
            file_path: base.join(format!("Upscaled/input96_{}.flac", format)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec![defect.to_string()],
            description: format!("{} transcoded to FLAC (should detect {} artifacts)", format.to_uppercase(), defect),
        });

        cases.push(TestCase {
            file_path: base.join(format!("Upscaled/input192_{}.flac", format)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec![defect.to_string()],
            description: format!("{} from 192kHz transcoded to FLAC (should detect {} artifacts)", format.to_uppercase(), defect),
        });
    }

    // MasterScript - Complex degradations (mostly fail except originals)
    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_original.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "MasterScript: Original 96kHz reference (should be clean)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("MasterScript/test192_original.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "MasterScript: Original 192kHz reference (should be clean)".to_string(),
    });

    // MasterScript MP3 artifacts
    let mp3_bitrates = vec!["128", "192", "256", "320"];
    for bitrate in mp3_bitrates {
        cases.push(TestCase {
            file_path: base.join(format!("MasterScript/test96_mp3_{}_upscaled.flac", bitrate)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec!["Mp3Transcode".to_string()],
            description: format!("MasterScript: MP3 {}k upscaled to FLAC (should detect MP3)", bitrate),
        });
    }

    // MasterScript bit depth upscaling
    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_16bit_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "MasterScript: 16-bit upscaled to 24-bit (should detect fake bit depth)".to_string(),
    });

    // MasterScript resampling artifacts
    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_resampled_44.1_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Upsampled".to_string()],
        description: "MasterScript: 44.1kHz upsampled to 192kHz (should detect upsampling)".to_string(),
    });

    // MasterScript generational loss
    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_mp3_320_reencoded_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "MasterScript: MP3→MP3 generational loss (should detect MP3)".to_string(),
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
        .output()
        .expect("Failed to execute binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Parse output to determine if defects were found
    let has_issues = stdout.contains("✗ ISSUES DETECTED") || 
                     stdout.contains("ISSUES DETECTED");
    
    let is_clean = stdout.contains("✓ CLEAN") || 
                   stdout.contains("CLEAN") && !has_issues;

    // Extract defect types
    let mut defects_found = Vec::new();
    
    if stdout.contains("MP3") || stdout.contains("Mp3") {
        defects_found.push("Mp3Transcode".to_string());
    }
    if stdout.contains("AAC") || stdout.contains("Aac") {
        defects_found.push("AacTranscode".to_string());
    }
    if stdout.contains("Opus") {
        defects_found.push("OpusTranscode".to_string());
    }
    if stdout.contains("Vorbis") || stdout.contains("Ogg") {
        defects_found.push("OggVorbisTranscode".to_string());
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
