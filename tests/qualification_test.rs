// tests/qualification_test.rs
// QUALIFICATION Test Suite - Compact subset for CI/CD quick validation
// Uses a subset of files from TestFiles/ for fast validation on every push
//
// Test Philosophy:
// - CleanOrigin: Original master files → PASS (genuine high-res)
// - CleanTranscoded: 24→16 bit honest transcodes → PASS (genuinely 16-bit)
// - Resample96: 96kHz → lower rates = PASS, 96kHz → higher rates = FAIL (interpolated)
// - Upscale16: 16-bit → 24-bit padding = FAIL (fake bit depth)
// - Upscaled: Lossy → Lossless = FAIL (lossy artifacts detected)

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
    #[allow(dead_code)]
    file: String,
}

/// Main qualification test - runs against TestFiles subset
#[test]
fn test_qualification_suite() {
    let binary_path = get_binary_path();
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let test_base = project_root.join("TestFiles");

    assert!(
        test_base.exists(),
        "TestFiles directory not found at: {}. \
         Download TestFiles.zip from MinIO for qualification tests.",
        test_base.display()
    );

    println!("\n{}", "=".repeat(60));
    println!("QUALIFICATION TEST SUITE");
    println!("Using: {}", test_base.display());
    println!("{}\n", "=".repeat(60));

    let test_cases = define_qualification_tests(&test_base);
    let mut passed = 0;
    let mut failed = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;

    println!("Running {} qualification tests...\n", test_cases.len());

    for (idx, test_case) in test_cases.iter().enumerate() {
        let result = run_test(&binary_path, test_case);

        if result.passed == result.expected {
            passed += 1;
            println!("[{:2}/{}] ✓ PASS: {}", idx + 1, test_cases.len(), test_case.description);
        } else {
            failed += 1;

            if result.passed && !result.expected {
                false_negatives += 1;
                println!("[{:2}/{}] ✗ FALSE NEGATIVE: {}", idx + 1, test_cases.len(), test_case.description);
                println!("        Expected defects but got CLEAN");
            } else {
                false_positives += 1;
                println!("[{:2}/{}] ✗ FALSE POSITIVE: {}", idx + 1, test_cases.len(), test_case.description);
                println!("        Expected CLEAN but detected defects: {:?}", result.defects_found);
            }
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("QUALIFICATION RESULTS");
    println!("{}", "=".repeat(60));
    println!("Total Tests:     {}", test_cases.len());
    println!("Passed:          {} ({:.1}%)", passed, (passed as f32 / test_cases.len() as f32) * 100.0);
    println!("Failed:          {}", failed);
    println!("  False Positives: {} (clean files marked as defective)", false_positives);
    println!("  False Negatives: {} (defective files marked as clean)", false_negatives);
    println!("{}", "=".repeat(60));

    assert_eq!(failed, 0, "Qualification failed: {} test(s) did not pass", failed);
}

fn get_binary_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");

    let release_path = path.join("release").join("audiocheckr");
    let debug_path = path.join("debug").join("audiocheckr");

    #[cfg(windows)]
    {
        let release_path_exe = release_path.with_extension("exe");
        let debug_path_exe = debug_path.with_extension("exe");

        if release_path_exe.exists() {
            return release_path_exe;
        } else if debug_path_exe.exists() {
            return debug_path_exe;
        }
    }

    #[cfg(unix)]
    {
        if release_path.exists() {
            return release_path;
        } else if debug_path.exists() {
            return debug_path;
        }
    }

    panic!("Binary not found. Run: cargo build --release");
}

fn define_qualification_tests(base: &Path) -> Vec<TestCase> {
    let mut cases = Vec::new();

    // =========================================================================
    // CLEANORIGIN - Original master files (should all PASS)
    // These are genuine high-resolution source files
    // =========================================================================
    
    cases.push(TestCase {
        file_path: base.join("CleanOrigin/input96.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "CleanOrigin: 96kHz 24-bit original master".to_string(),
    });

    // Note: input192.flac is 16-bit source, so it should fail bit depth check when
    // claiming to be 24-bit. If the file genuinely has 24-bit samples, change to true.
    cases.push(TestCase {
        file_path: base.join("CleanOrigin/input192.flac").to_string_lossy().to_string(),
        should_pass: false,  // 16-bit source upscaled to 24-bit container
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "CleanOrigin: 192kHz (16-bit source in 24-bit container)".to_string(),
    });

    // =========================================================================
    // CLEANTRANSCODED - Honest bit depth reduction (should PASS)
    // 24-bit → 16-bit transcode, genuinely 16-bit data in 16-bit container
    // =========================================================================
    
    cases.push(TestCase {
        file_path: base.join("CleanTranscoded/input96_16bit.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "CleanTranscoded: 96kHz honest 16-bit transcode".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("CleanTranscoded/input192_16bit.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "CleanTranscoded: 192kHz honest 16-bit transcode".to_string(),
    });

    // =========================================================================
    // RESAMPLE96 - Sample rate changes from 96kHz source
    // Downsampling (96→lower) = PASS, Upsampling (96→higher) = FAIL
    // =========================================================================
    
    // Downsample: 96kHz → 44.1kHz (genuine, no interpolation) → PASS
    cases.push(TestCase {
        file_path: base.join("Resample96/input96_44.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "Resample96: 96→44.1kHz downsampled (genuine)".to_string(),
    });

    // Downsample: 96kHz → 48kHz (genuine) → PASS
    cases.push(TestCase {
        file_path: base.join("Resample96/input96_48.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "Resample96: 96→48kHz downsampled (genuine)".to_string(),
    });

    // Downsample: 96kHz → 88.2kHz (genuine) → PASS
    cases.push(TestCase {
        file_path: base.join("Resample96/input96_88.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "Resample96: 96→88.2kHz downsampled (genuine)".to_string(),
    });

    // Upsample: 96kHz → 176.4kHz (interpolated data) → FAIL
    cases.push(TestCase {
        file_path: base.join("Resample96/input96_176.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Upsampled".to_string()],
        description: "Resample96: 96→176.4kHz upsampled (interpolated)".to_string(),
    });

    // Upsample: 96kHz → 192kHz (interpolated data) → FAIL
    cases.push(TestCase {
        file_path: base.join("Resample96/input96_192.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Upsampled".to_string()],
        description: "Resample96: 96→192kHz upsampled (interpolated)".to_string(),
    });

    // =========================================================================
    // UPSCALE16 - 16-bit to 24-bit padding (should FAIL)
    // Fake bit depth: zero-padded LSBs
    // =========================================================================
    
    cases.push(TestCase {
        file_path: base.join("Upscale16/output96_16bit.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "Upscale16: 96kHz 16→24-bit upscaled (fake depth)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscale16/output192_16bit.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "Upscale16: 192kHz 16→24-bit upscaled (fake depth)".to_string(),
    });

    // =========================================================================
    // UPSCALED - Lossy codec transcodes to FLAC (should FAIL)
    // Each codec has characteristic artifacts
    // =========================================================================
    
    // MP3 transcodes
    cases.push(TestCase {
        file_path: base.join("Upscaled/input96_mp3.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Upscaled: 96kHz from MP3 (lossy artifacts)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscaled/input192_mp3.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Upscaled: 192kHz from MP3 (lossy artifacts)".to_string(),
    });

    // AAC transcodes
    cases.push(TestCase {
        file_path: base.join("Upscaled/input96_m4a.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["AacTranscode".to_string()],
        description: "Upscaled: 96kHz from AAC/M4A (lossy artifacts)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscaled/input192_m4a.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["AacTranscode".to_string()],
        description: "Upscaled: 192kHz from AAC/M4A (lossy artifacts)".to_string(),
    });

    // Opus transcodes
    cases.push(TestCase {
        file_path: base.join("Upscaled/input96_opus.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OpusTranscode".to_string()],
        description: "Upscaled: 96kHz from Opus (lossy artifacts)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscaled/input192_opus.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OpusTranscode".to_string()],
        description: "Upscaled: 192kHz from Opus (lossy artifacts)".to_string(),
    });

    // Vorbis transcodes
    cases.push(TestCase {
        file_path: base.join("Upscaled/input96_ogg.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OggVorbisTranscode".to_string()],
        description: "Upscaled: 96kHz from Vorbis/OGG (lossy artifacts)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscaled/input192_ogg.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OggVorbisTranscode".to_string()],
        description: "Upscaled: 192kHz from Vorbis/OGG (lossy artifacts)".to_string(),
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

    let has_issues = stdout.contains("✗ ISSUES DETECTED") || stdout.contains("ISSUES DETECTED");
    let is_clean = stdout.contains("✓ CLEAN") || (stdout.contains("CLEAN") && !has_issues);

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
    if stdout.contains("Bit depth mismatch") || stdout.contains("BitDepth") || stdout.contains("bit depth") {
        defects_found.push("BitDepthMismatch".to_string());
    }
    if stdout.contains("Upsampled") || stdout.contains("upsampled") || stdout.contains("interpolat") {
        defects_found.push("Upsampled".to_string());
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
        .expect("Failed to run --help");
    
    assert!(output.status.success(), "Help command failed");
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("audiocheckr"), "Help output should mention audiocheckr");
}
