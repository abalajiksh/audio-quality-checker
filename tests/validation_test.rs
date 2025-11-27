// tests/validation_test.rs
// GROUND TRUTH Validation Test Suite (focused CI/CD subset)

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
fn test_all_audio_files() {
    let binary_path = get_binary_path();
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let test_base = project_root.join("TestFiles");

    assert!(test_base.exists());

    println!("\n=== Validation Tests ===\n");

    let test_cases = define_test_cases(&test_base);
    let mut results = Vec::new();
    let mut passed = 0;
    let mut failed = 0;

    for test_case in &test_cases {
        let result = run_test(&binary_path, test_case);

        if result.passed == result.expected {
            passed += 1;
            println!("✓ PASS: {}", test_case.description);
        } else {
            failed += 1;
            println!("✗ FAIL: {}", test_case.description);
            println!("  Expected: {}, Got: {}",
                if test_case.should_pass { "CLEAN" } else { "DEFECTS" },
                if result.passed { "CLEAN" } else { "DEFECTS" });
        }

        results.push(result);
    }

    println!("\n=== Summary ===");
    println!("Passed: {}/{} ({:.1}%)", passed, test_cases.len(), 
        (passed as f32 / test_cases.len() as f32) * 100.0);

    assert_eq!(failed, 0, "{} test(s) failed", failed);
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

    panic!("Binary not found");
}

fn define_test_cases(base: &Path) -> Vec<TestCase> {
    let mut cases = Vec::new();

    // CleanOrigin
    cases.push(TestCase {
        file_path: base.join("CleanOrigin/input96.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "96kHz 24-bit original (clean)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("CleanOrigin/input192.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "192kHz original (16-bit source)".to_string(),
    });

    // CleanTranscoded
    cases.push(TestCase {
        file_path: base.join("CleanTranscoded/input96_16bit.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "Honest 16-bit from 96kHz".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("CleanTranscoded/input192_16bit.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "Honest 16-bit from 192kHz".to_string(),
    });

    // Resample96
    for rate in ["44", "48", "88"] {
        cases.push(TestCase {
            file_path: base.join(format!("Resample96/input96_{}.flac", rate)).to_string_lossy().to_string(),
            should_pass: true,
            expected_defects: vec![],
            description: format!("Clean resample 96→{}kHz", rate),
        });
    }

    // Resample192
    for rate in ["44", "48", "88", "96", "176"] {
        cases.push(TestCase {
            file_path: base.join(format!("Resample192/input192_{}.flac", rate)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec!["BitDepthMismatch".to_string()],
            description: format!("Resample 192→{}kHz (16-bit)", rate),
        });
    }

    // Upscale16
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
        description: "16-bit upscaled to 24-bit (192)".to_string(),
    });

    // Upscaled
    cases.push(TestCase {
        file_path: base.join("Upscaled/input96_mp3.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "MP3 from 96kHz".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscaled/input192_mp3.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "MP3 from 192kHz".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscaled/input192_m4a.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["AacTranscode".to_string()],
        description: "AAC from 192kHz".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscaled/input192_opus.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OpusTranscode".to_string()],
        description: "Opus from 192kHz".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscaled/input192_ogg.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OggVorbisTranscode".to_string()],
        description: "Vorbis from 192kHz".to_string(),
    });

    // MasterScript
    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_original.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "96kHz original (clean)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("MasterScript/test192_original.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "192kHz original (16-bit)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_16bit_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "96kHz 16-bit upscaled".to_string(),
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
        .expect("Failed to execute");

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
    if stdout.contains("Bit depth mismatch") || stdout.contains("BitDepth") {
        defects_found.push("BitDepthMismatch".to_string());
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
    assert!(binary_path.exists());
}

#[test]
fn test_help_output() {
    let binary_path = get_binary_path();
    let output = Command::new(&binary_path).arg("--help").output().expect("Failed");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("audiocheckr"));
}
//colinsucks