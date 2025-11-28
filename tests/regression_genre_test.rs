// tests/regression_genre_tests.rs

// REGRESSION Genre Test Suite - Full TestSuite directory
// Comprehensive genre testing for weekly validation
//
// Purpose: Full validation across all genres and defect categories
// - Includes edge cases and complex transcoding chains
// - Multi-generation transcodes (MP3→MP3, Opus→MP3)
// - Sample rate manipulations (44→96kHz, 48→96kHz)
// - Low bitrate extremes (Opus 48k, MP3 64k)
// - Multi-stage resampling

use std::env;
use std::process::Command;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Clone)]
struct GenreTestCase {
    file_path: String,
    should_pass: bool,
    expected_defects: Vec<String>,
    description: String,
    genre: String,
    defect_category: String,
}

#[derive(Debug)]
struct TestResult {
    passed: bool,
    expected: bool,
    defects_found: Vec<String>,
    description: String,
    genre: String,
    #[allow(dead_code)]
    file: String,
}

/// Main regression genre test
#[test]
fn test_regression_genre_suite() {
    let binary_path = get_binary_path();
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let test_base = project_root.join("TestSuite");
    
    assert!(
        test_base.exists(),
        "TestSuite directory not found at: {}. \
        Download TestSuite.zip from MinIO for regression tests.",
        test_base.display()
    );
    
    println!("\n{}", "=".repeat(60));
    println!("REGRESSION GENRE TEST SUITE (Full Validation)");
    println!("Using: {}", test_base.display());
    println!("{}\n", "=".repeat(60));
    
    let test_cases = define_regression_genre_tests(&test_base);
    let total_tests = test_cases.len();
    
    println!("Running {} comprehensive genre tests in parallel...\n", total_tests);
    
    // Run tests in parallel with 6 threads (more intensive test suite)
    let results = run_tests_parallel(&binary_path, test_cases, 6);
    
    // Analyze results by genre and category
    analyze_results(&results, total_tests);
}

fn analyze_results(results: &[TestResult], total_tests: usize) {
    let mut passed = 0;
    let mut failed = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;
    
    // Print individual results
    for (idx, result) in results.iter().enumerate() {
        if result.passed == result.expected {
            passed += 1;
            println!(
                "[{:3}/{}] ✓ PASS [{}]: {}", 
                idx + 1, total_tests, result.genre, result.description
            );
        } else {
            failed += 1;
            if result.passed && !result.expected {
                false_negatives += 1;
                println!(
                    "[{:3}/{}] ✗ FALSE NEGATIVE [{}]: {}", 
                    idx + 1, total_tests, result.genre, result.description
                );
                println!("     Expected defects but got CLEAN");
            } else {
                false_positives += 1;
                println!(
                    "[{:3}/{}] ✗ FALSE POSITIVE [{}]: {}", 
                    idx + 1, total_tests, result.genre, result.description
                );
                println!("     Expected CLEAN but detected: {:?}", result.defects_found);
            }
        }
    }
    
    println!("\n{}", "=".repeat(60));
    println!("REGRESSION GENRE RESULTS");
    println!("{}", "=".repeat(60));
    println!("Total Tests: {}", total_tests);
    println!("Passed: {} ({:.1}%)", passed, (passed as f32 / total_tests as f32) * 100.0);
    println!("Failed: {}", failed);
    println!("  False Positives: {} (clean marked as defective)", false_positives);
    println!("  False Negatives: {} (defective marked as clean)", false_negatives);
    println!("{}", "=".repeat(60));
    
    assert_eq!(failed, 0, "Regression genre tests failed: {} test(s) did not pass", failed);
}

/// Run tests in parallel
fn run_tests_parallel(binary: &Path, test_cases: Vec<GenreTestCase>, num_threads: usize) -> Vec<TestResult> {
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
                let current_idx = {
                    let mut idx = index.lock().unwrap();
                    if *idx >= test_cases.len() {
                        return;
                    }
                    let current = *idx;
                    *idx += 1;
                    current
                };
                
                let test_case = &test_cases[current_idx];
                let result = run_single_test(&binary, test_case);
                
                let mut results_guard = results.lock().unwrap();
                results_guard.push((current_idx, result));
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
    
    let mut results_vec: Vec<(usize, TestResult)> = Arc::try_unwrap(results)
        .expect("Arc still has multiple owners")
        .into_inner()
        .expect("Mutex poisoned");
    results_vec.sort_by_key(|(idx, _)| *idx);
    results_vec.into_iter().map(|(_, result)| result).collect()
}

fn run_single_test(binary: &Path, test_case: &GenreTestCase) -> TestResult {
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
        genre: test_case.genre.clone(),
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

fn define_regression_genre_tests(base: &Path) -> Vec<GenreTestCase> {
    let mut cases = Vec::new();
    
    // =========================================================================
    // EDGE CASES - Low bitrate extremes
    // =========================================================================
    
    // Alternative - Opus 48k extreme low
    cases.push(GenreTestCase {
        file_path: base.join("Edge_LowBitrate_Opus/Instant_Destiny_24sample_opus_48k_extreme.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OpusTranscode".to_string()],
        description: "Instant Destiny - Opus 48k (extreme low)".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "EdgeCase_LowBitrate".to_string(),
    });
    
    // Alternative - Opus 48k extreme low (16-bit)
    cases.push(GenreTestCase {
        file_path: base.join("Edge_LowBitrate_Opus/Paranoid_Android__Remastered__16sample_opus_48k_extreme.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OpusTranscode".to_string()],
        description: "Paranoid Android - Opus 48k (extreme low)".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "EdgeCase_LowBitrate".to_string(),
    });
    
    // Alternative - MP3 64k extreme low
    cases.push(GenreTestCase {
        file_path: base.join("Edge_MP3_64k/Instant_Destiny_24sample_mp3_64k_extreme.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Instant Destiny - MP3 64k (extreme low)".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "EdgeCase_MP3_64k".to_string(),
    });
    
    // =========================================================================
    // MULTI-GENERATION TRANSCODES
    // =========================================================================
    
    // Alternative - MP3 → MP3 (generation 2)
    cases.push(GenreTestCase {
        file_path: base.join("Generation_MP3_MP3/Instant_Destiny_24sample_mp3_gen2.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Instant Destiny - MP3→MP3 (2nd gen)".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "MultiGeneration".to_string(),
    });
    
    // Alternative - MP3 → MP3 (generation 2, 16-bit)
    cases.push(GenreTestCase {
        file_path: base.join("Generation_MP3_MP3/Paranoid_Android__Remastered__16sample_mp3_gen2.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Paranoid Android - MP3→MP3 (2nd gen)".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "MultiGeneration".to_string(),
    });
    
    // Alternative - Opus → MP3 cross-codec
    cases.push(GenreTestCase {
        file_path: base.join("Generation_Opus_MP3/Instant_Destiny_24sample_opus_to_mp3.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Instant Destiny - Opus→MP3".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "CrossCodec".to_string(),
    });
    
    // =========================================================================
    // SAMPLE RATE UPSAMPLING
    // =========================================================================
    
    // Alternative - 44.1→96kHz
    cases.push(GenreTestCase {
        file_path: base.join("SampleRate_44to96/Instant_Destiny_24sample_44to96_upsampled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Upsampled".to_string()],
        description: "Instant Destiny - 44.1→96kHz upsampled".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "SampleRateUpsample".to_string(),
    });
    
    // Electronic - 48→96kHz
    cases.push(GenreTestCase {
        file_path: base.join("SampleRate_48to96/Different_Masks_For_Different_Days__Live_from_Echostage__Washington__24sample_48to96_upsampled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Upsampled".to_string()],
        description: "Different Masks - 48→96kHz upsampled".to_string(),
        genre: "ElectronicDance".to_string(),
        defect_category: "SampleRateUpsample".to_string(),
    });
    
    // Ambient - 48→96kHz
    cases.push(GenreTestCase {
        file_path: base.join("SampleRate_48to96/inconsist_24sample_48to96_upsampled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Upsampled".to_string()],
        description: "inconsist - 48→96kHz upsampled".to_string(),
        genre: "AmbientDrone".to_string(),
        defect_category: "SampleRateUpsample".to_string(),
    });
    
    // Alternative - 48→96kHz
    cases.push(GenreTestCase {
        file_path: base.join("SampleRate_48to96/Instant_Destiny_24sample_48to96_upsampled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Upsampled".to_string()],
        description: "Instant Destiny - 48→96kHz upsampled".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "SampleRateUpsample".to_string(),
    });
    
    // =========================================================================
    // MULTI-STAGE RESAMPLING
    // =========================================================================
    
    // Classical - Multiple resampling stages
    cases.push(GenreTestCase {
        file_path: base.join("Edge_MultipleResample/Brandenburg_Concerto_in_G_MajorBWV_1049_16sample_multi_resample.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Upsampled".to_string()],
        description: "Brandenburg Concerto - Multi-stage resample".to_string(),
        genre: "Classical".to_string(),
        defect_category: "MultiResample".to_string(),
    });
    
    // Folk - Multiple resampling stages
    cases.push(GenreTestCase {
        file_path: base.join("Edge_MultipleResample/We_24sample_multi_resample.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Upsampled".to_string()],
        description: "We - Multi-stage resample".to_string(),
        genre: "Folk".to_string(),
        defect_category: "MultiResample".to_string(),
    });
    
    // =========================================================================
    // COMBINED DEFECTS (BitDepth + SampleRate)
    // =========================================================================
    
    // Ambient - 16-bit + 44kHz upsampled
    cases.push(GenreTestCase {
        file_path: base.join("Combined_16bit_44khz/inconsist_24sample_16bit_44khz_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string(), "Upsampled".to_string()],
        description: "inconsist - 16-bit + 44kHz upscaled".to_string(),
        genre: "AmbientDrone".to_string(),
        defect_category: "Combined".to_string(),
    });
    
    // Alternative - 16-bit + 44kHz upsampled
    cases.push(GenreTestCase {
        file_path: base.join("Combined_16bit_44khz/Instant_Destiny_24sample_16bit_44khz_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string(), "Upsampled".to_string()],
        description: "Instant Destiny - 16-bit + 44kHz upscaled".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "Combined".to_string(),
    });
    
    // =========================================================================
    // COMBINED DEFECTS (CD → MP3 → 24-bit upscale)
    // =========================================================================
    
    // Electronic - CD → MP3 128 → 24-bit
    cases.push(GenreTestCase {
        file_path: base.join("Combined_MP3_128_From_CD/Different_Masks_For_Different_Days__Live_from_Echostage__Washington__24sample_cd_mp3_128k_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string(), "BitDepthMismatch".to_string()],
        description: "Different Masks - CD→MP3→24bit".to_string(),
        genre: "ElectronicDance".to_string(),
        defect_category: "Combined".to_string(),
    });
    
    // Ambient - CD → MP3 128 → 24-bit
    cases.push(GenreTestCase {
        file_path: base.join("Combined_MP3_128_From_CD/inconsist_24sample_cd_mp3_128k_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string(), "BitDepthMismatch".to_string()],
        description: "inconsist - CD→MP3→24bit".to_string(),
        genre: "AmbientDrone".to_string(),
        defect_category: "Combined".to_string(),
    });
    
    // =========================================================================
    // VORBIS TRANSCODES
    // =========================================================================
    
    // Alternative - Vorbis Q3 (low quality)
    cases.push(GenreTestCase {
        file_path: base.join("Vorbis_Q3_Low/Instant_Destiny_24sample_vorbis_q3.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OggVorbisTranscode".to_string()],
        description: "Instant Destiny - Vorbis Q3".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "Vorbis_Q3".to_string(),
    });
    
    // Alternative - Vorbis Q3 (16-bit)
    cases.push(GenreTestCase {
        file_path: base.join("Vorbis_Q3_Low/Paranoid_Android__Remastered__16sample_vorbis_q3.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OggVorbisTranscode".to_string()],
        description: "Paranoid Android - Vorbis Q3".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "Vorbis_Q3".to_string(),
    });
    
    // Alternative - Vorbis Q7 (high quality)
    cases.push(GenreTestCase {
        file_path: base.join("Vorbis_Q7_High/Instant_Destiny_24sample_vorbis_q7.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OggVorbisTranscode".to_string()],
        description: "Instant_Destiny - Vorbis Q7".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "Vorbis_Q7".to_string(),
    });
    
    // Alternative - Vorbis Q7 (16-bit)
    cases.push(GenreTestCase {
        file_path: base.join("Vorbis_Q7_High/Paranoid_Android__Remastered__16sample_vorbis_q7.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OggVorbisTranscode".to_string()],
        description: "Paranoid Android - Vorbis Q7".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "Vorbis_Q7".to_string(),
    });
    
    // Total: 3 edge + 3 multi-gen + 4 sample rate + 2 multi-resample + 4 combined + 4 vorbis = 20 tests
    
    cases
}
