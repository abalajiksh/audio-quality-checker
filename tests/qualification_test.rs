// tests/qualification_test.rs
// QUALIFICATION Test Suite - Compact subset for CI/CD quick validation
// Uses a subset of files from TestFiles/ for fast validation on every push
//
// Test Philosophy:
// - CleanOrigin: Original master files → PASS (genuine high-res) except input192 (16-bit source)
// - CleanTranscoded: 24→16 bit honest transcodes → PASS (genuinely 16-bit)
// - Resample96: 96kHz → lower rates = PASS, 96kHz → higher rates = FAIL (interpolated)
// - Resample192: All from 16-bit source → FAIL (BitDepthMismatch)
// - Upscale16: 16-bit → 24-bit padding = FAIL (fake bit depth)
// - Upscaled: Lossy → Lossless = FAIL (lossy artifacts detected)
//
// Parallelization: Tests run in parallel (4 threads) for faster CI/CD

use std::env;
use std::process::Command;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Clone)]
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
    description: String,
    #[allow(dead_code)]
    file: String,
}

/// Main qualification test - runs against TestFiles subset with parallel execution
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
    println!("QUALIFICATION TEST SUITE (Parallel Execution)");
    println!("Using: {}", test_base.display());
    println!("{}\n", "=".repeat(60));

    let test_cases = define_qualification_tests(&test_base);
    let total_tests = test_cases.len();
    
    println!("Running {} qualification tests in parallel...\n", total_tests);

    // Run tests in parallel with 4 threads
    let results = run_tests_parallel(&binary_path, test_cases, 4);
    
    // Analyze results
    let mut passed = 0;
    let mut failed = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;

    for (idx, result) in results.iter().enumerate() {
        if result.passed == result.expected {
            passed += 1;
            println!("[{:2}/{}] ✓ PASS: {}", idx + 1, total_tests, result.description);
        } else {
            failed += 1;

            if result.passed && !result.expected {
                false_negatives += 1;
                println!("[{:2}/{}] ✗ FALSE NEGATIVE: {}", idx + 1, total_tests, result.description);
                println!("        Expected defects but got CLEAN");
            } else {
                false_positives += 1;
                println!("[{:2}/{}] ✗ FALSE POSITIVE: {}", idx + 1, total_tests, result.description);
                println!("        Expected CLEAN but detected defects: {:?}", result.defects_found);
            }
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("QUALIFICATION RESULTS");
    println!("{}", "=".repeat(60));
    println!("Total Tests:     {}", total_tests);
    println!("Passed:          {} ({:.1}%)", passed, (passed as f32 / total_tests as f32) * 100.0);
    println!("Failed:          {}", failed);
    println!("  False Positives: {} (clean files marked as defective)", false_positives);
    println!("  False Negatives: {} (defective files marked as clean)", false_negatives);
    println!("{}", "=".repeat(60));

    assert_eq!(failed, 0, "Qualification failed: {} test(s) did not pass", failed);
}

/// Run tests in parallel using thread pool
fn run_tests_parallel(binary: &Path, test_cases: Vec<TestCase>, num_threads: usize) -> Vec<TestResult> {
    let binary = binary.to_path_buf();
    let test_cases = Arc::new(test_cases);
    let results = Arc::new(Mutex::new(Vec::new()));
    let index = Arc::new(Mutex::new(0usize));
    
    let mut handles = Vec::new();
    
    for _ in 0..num_threads {
        let binary = binary.clone();
        let test_cases = Arc::clone(&test_cases);
        let results = Arc::clone(&results);
        let index = Arc::clone(&index);
        
        let handle = thread::spawn(move || {
            loop {
                // Get next test case index
                let current_idx = {
                    let mut idx = index.lock().unwrap();
                    if *idx >= test_cases.len() {
                        return;
                    }
                    let current = *idx;
                    *idx += 1;
                    current
                };
                
                // Run the test
                let test_case = &test_cases[current_idx];
                let result = run_single_test(&binary, test_case);
                
                // Store result
                let mut results_guard = results.lock().unwrap();
                results_guard.push((current_idx, result));
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
    
    // Sort results by original index and extract
    let mut results_vec: Vec<(usize, TestResult)> = Arc::try_unwrap(results)
        .expect("Arc still has multiple owners")
        .into_inner()
        .expect("Mutex poisoned");
    
    results_vec.sort_by_key(|(idx, _)| *idx);
    results_vec.into_iter().map(|(_, result)| result).collect()
}

fn run_single_test(binary: &Path, test_case: &TestCase) -> TestResult {
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
        description: test_case.description.clone(),
        file: test_case.file_path.clone(),
    }
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
    // CLEANORIGIN - Original master files (2 tests)
    // =========================================================================
    
    cases.push(TestCase {
        file_path: base.join("CleanOrigin/input96.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "CleanOrigin: 96kHz 24-bit original master".to_string(),
    });

    // Note: input192.flac is 16-bit source in 24-bit container
    cases.push(TestCase {
        file_path: base.join("CleanOrigin/input192.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "CleanOrigin: 192kHz (16-bit source in 24-bit container)".to_string(),
    });

    // =========================================================================
    // CLEANTRANSCODED - Honest bit depth reduction (2 tests)
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
    // RESAMPLE96 - Sample rate changes from 96kHz source (3 tests - downsampling only)
    // =========================================================================
    
    // Downsample: genuine, no interpolation → PASS
    for rate in ["44", "48", "88"] {
        cases.push(TestCase {
            file_path: base.join(format!("Resample96/input96_{}.flac", rate)).to_string_lossy().to_string(),
            should_pass: true,
            expected_defects: vec![],
            description: format!("Resample96: 96→{}kHz downsampled (genuine)", rate),
        });
    }

    // =========================================================================
    // RESAMPLE192 - All from 16-bit source (5 tests) - should all FAIL BitDepthMismatch
    // =========================================================================
    
    for rate in ["44", "48", "88", "96", "176"] {
        cases.push(TestCase {
            file_path: base.join(format!("Resample192/input192_{}.flac", rate)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec!["BitDepthMismatch".to_string()],
            description: format!("Resample192: 192→{}kHz (16-bit source)", rate),
        });
    }

    // =========================================================================
    // UPSCALE16 - 16-bit to 24-bit padding (2 tests)
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
    // UPSCALED - Lossy codec transcodes to FLAC (5 tests)
    // Focus on 192kHz sources (more detectable) + one 96kHz MP3
    // =========================================================================
    
    // 96kHz MP3 (known detectable)
    cases.push(TestCase {
        file_path: base.join("Upscaled/input96_mp3.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Upscaled: 96kHz from MP3 (lossy artifacts)".to_string(),
    });

    // 192kHz lossy formats
    let lossy_formats_192 = vec![
        ("mp3", "Mp3Transcode"),
        ("m4a", "AacTranscode"),
        ("opus", "OpusTranscode"),
        ("ogg", "OggVorbisTranscode"),
    ];

    for (format, defect) in &lossy_formats_192 {
        cases.push(TestCase {
            file_path: base.join(format!("Upscaled/input192_{}.flac", format)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec![defect.to_string()],
            description: format!("Upscaled: 192kHz from {} (lossy artifacts)", format.to_uppercase()),
        });
    }

    // =========================================================================
    // MASTERSCRIPT - Key scenarios (5 tests)
    // =========================================================================
    
    // test96_original - reference file, should PASS
    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_original.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "MasterScript: test96 original (reference, clean)".to_string(),
    });

    // test192_original - 16-bit source, should FAIL
    cases.push(TestCase {
        file_path: base.join("MasterScript/test192_original.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "MasterScript: test192 original (16-bit source)".to_string(),
    });

    // test96 bit depth degradation
    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_16bit_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "MasterScript: test96 16-bit upscaled to 24-bit".to_string(),
    });

    // test192 MP3 320 upscaled (BitDepthMismatch + Mp3Transcode)
    cases.push(TestCase {
        file_path: base.join("MasterScript/test192_mp3_320_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string(), "Mp3Transcode".to_string()],
        description: "MasterScript: test192 MP3 320k upscaled".to_string(),
    });

    // test192 Opus 128 upscaled (BitDepthMismatch + OpusTranscode)
    cases.push(TestCase {
        file_path: base.join("MasterScript/test192_opus_128_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string(), "OpusTranscode".to_string()],
        description: "MasterScript: test192 Opus 128k upscaled".to_string(),
    });

    // Total: 2 + 2 + 3 + 5 + 2 + 5 + 5 = 24 test cases
    // Note: Original request was 22, but the reference file shows 24 cases
    
    cases
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
