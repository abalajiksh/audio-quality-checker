// tests/qualification_genre_tests.rs

// QUALIFICATION Genre Test Suite - GenreTestSuiteLite (45 files)
// Based on manifest.txt - Generated: 2025-11-28 15:37:12
//
// Purpose: Quick qualification test for audiocheckr with genre-representative files
// - Each test is labeled with genre for easier debugging
// - Covers all major defect categories
// - Runs in parallel (4 threads) for faster CI/CD

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

#[test]
fn test_qualification_genre_suite() {
    let binary_path = get_binary_path();
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let test_base = project_root.join("GenreTestSuiteLite");
    
    assert!(
        test_base.exists(),
        "GenreTestSuiteLite directory not found at: {}. \
        Download GenreTestSuiteLite.zip from MinIO.",
        test_base.display()
    );
    
    println!("\n{}", "=".repeat(70));
    println!("QUALIFICATION GENRE TEST SUITE (45 files - Parallel Execution)");
    println!("Using: {}", test_base.display());
    println!("{}\n", "=".repeat(70));
    
    let test_cases = define_qualification_genre_tests(&test_base);
    let total_tests = test_cases.len();
    
    println!("Running {} genre qualification tests in parallel...\n", total_tests);
    
    let results = run_tests_parallel(&binary_path, test_cases, 4);
    
    let mut passed = 0;
    let mut failed = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;
    
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
    
    println!("\n{}", "=".repeat(70));
    println!("QUALIFICATION GENRE RESULTS");
    println!("{}", "=".repeat(70));
    println!("Total Tests: {}", total_tests);
    println!("Passed: {} ({:.1}%)", passed, (passed as f32 / total_tests as f32) * 100.0);
    println!("Failed: {}", failed);
    println!("  False Positives: {} (clean marked as defective)", false_positives);
    println!("  False Negatives: {} (defective marked as clean)", false_negatives);
    println!("{}", "=".repeat(70));
    
    assert_eq!(failed, 0, "Qualification genre tests failed: {} test(s) did not pass", failed);
}

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

fn define_qualification_genre_tests(base: &Path) -> Vec<GenreTestCase> {
    let mut cases = Vec::new();
    
    // =========================================================================
    // CONTROL_ORIGINAL - Authentic files (should PASS) - 2 files
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("Control_Original/Exile_24sample_control.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "Exile - 24-bit authentic".to_string(),
        genre: "Folk".to_string(),
        defect_category: "Control_Original".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("Control_Original/Pride_and_Joy_16sample_control.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "Pride and Joy - 16-bit authentic".to_string(),
        genre: "Blues".to_string(),
        defect_category: "Control_Original".to_string(),
    });
    
    // =========================================================================
    // AAC_128_Low - AAC 128kbps (should FAIL) - 2 files
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("AAC_128_Low/Different_Masks_For_Different_Days__Live_from_Echostage__Washington__24sample_aac_128k.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["AacTranscode".to_string()],
        description: "Different Masks - AAC 128k".to_string(),
        genre: "ElectronicDance".to_string(),
        defect_category: "AAC_128_Low".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("AAC_128_Low/inconsist_24sample_aac_128k.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["AacTranscode".to_string()],
        description: "inconsist - AAC 128k".to_string(),
        genre: "AmbientDrone".to_string(),
        defect_category: "AAC_128_Low".to_string(),
    });
    
    // =========================================================================
    // AAC_256_High - AAC 256kbps (should FAIL) - 3 files
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("AAC_256_High/Could_You_Be_Loved__Album_Version__16sample_aac_256k.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["AacTranscode".to_string()],
        description: "Could You Be Loved - AAC 256k".to_string(),
        genre: "ReggaeDub".to_string(),
        defect_category: "AAC_256_High".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("AAC_256_High/MALAMENTE__Cap_1_-__Augurio__24sample_aac_256k.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["AacTranscode".to_string()],
        description: "MALAMENTE - AAC 256k".to_string(),
        genre: "LatinWorld".to_string(),
        defect_category: "AAC_256_High".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("AAC_256_High/Wake_Up_16sample_aac_256k.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["AacTranscode".to_string()],
        description: "Wake Up - AAC 256k".to_string(),
        genre: "Indie".to_string(),
        defect_category: "AAC_256_High".to_string(),
    });
    
    // =========================================================================
    // BitDepth_16to24 - 16→24 bit padding (should FAIL) - 4 files
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("BitDepth_16to24/Boogieman_24sample_16to24_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "Boogieman - 16→24 bit padding".to_string(),
        genre: "HipHopRnB".to_string(),
        defect_category: "BitDepth_16to24".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("BitDepth_16to24/Dream_of_Arrakis_24sample_16to24_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "Dream of Arrakis - 16→24 bit padding".to_string(),
        genre: "SoundtrackScore".to_string(),
        defect_category: "BitDepth_16to24".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("BitDepth_16to24/Exile_24sample_16to24_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "Exile - 16→24 bit padding".to_string(),
        genre: "Folk".to_string(),
        defect_category: "BitDepth_16to24".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("BitDepth_16to24/Punisher_24sample_16to24_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "Punisher - 16→24 bit padding".to_string(),
        genre: "Indie".to_string(),
        defect_category: "BitDepth_16to24".to_string(),
    });
    
    // =========================================================================
    // Combined_16bit_44khz - BitDepth + SampleRate (should FAIL) - 2 files
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("Combined_16bit_44khz/inconsist_24sample_16bit_44khz_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string(), "Upsampled".to_string()],
        description: "inconsist - 16-bit + 44kHz→96kHz".to_string(),
        genre: "AmbientDrone".to_string(),
        defect_category: "Combined_16bit_44khz".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("Combined_16bit_44khz/Instant_Destiny_24sample_16bit_44khz_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string(), "Upsampled".to_string()],
        description: "Instant Destiny - 16-bit + 44kHz→96kHz".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "Combined_16bit_44khz".to_string(),
    });
    
    // =========================================================================
    // Combined_MP3_128_From_CD - CD→MP3→24bit (should FAIL) - 2 files
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("Combined_MP3_128_From_CD/Different_Masks_For_Different_Days__Live_from_Echostage__Washington__24sample_cd_mp3_128k_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string(), "BitDepthMismatch".to_string()],
        description: "Different Masks - CD→MP3 128k→24bit".to_string(),
        genre: "ElectronicDance".to_string(),
        defect_category: "Combined_MP3_128_From_CD".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("Combined_MP3_128_From_CD/inconsist_24sample_cd_mp3_128k_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string(), "BitDepthMismatch".to_string()],
        description: "inconsist - CD→MP3 128k→24bit".to_string(),
        genre: "AmbientDrone".to_string(),
        defect_category: "Combined_MP3_128_From_CD".to_string(),
    });
    
    // =========================================================================
    // Edge_LowBitrate_Opus - Opus 48kbps extreme (should FAIL) - 2 files
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("Edge_LowBitrate_Opus/Instant_Destiny_24sample_opus_48k_extreme.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OpusTranscode".to_string()],
        description: "Instant Destiny - Opus 48k extreme".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "Edge_LowBitrate_Opus".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("Edge_LowBitrate_Opus/Paranoid_Android__Remastered__16sample_opus_48k_extreme.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OpusTranscode".to_string()],
        description: "Paranoid Android - Opus 48k extreme".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "Edge_LowBitrate_Opus".to_string(),
    });
    
    // =========================================================================
    // Edge_MP3_64k - MP3 64kbps extreme (should FAIL) - 1 file
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("Edge_MP3_64k/Instant_Destiny_24sample_mp3_64k_extreme.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Instant Destiny - MP3 64k extreme".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "Edge_MP3_64k".to_string(),
    });
    
    // =========================================================================
    // Edge_MultipleResample - Multi-stage resampling (should FAIL) - 2 files
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("Edge_MultipleResample/Brandenburg_Concerto_in_G_MajorBWV_1049_16sample_multi_resample.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Upsampled".to_string()],
        description: "Brandenburg Concerto - Multi-resample".to_string(),
        genre: "Classical".to_string(),
        defect_category: "Edge_MultipleResample".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("Edge_MultipleResample/We_24sample_multi_resample.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Upsampled".to_string()],
        description: "We - Multi-resample".to_string(),
        genre: "Folk".to_string(),
        defect_category: "Edge_MultipleResample".to_string(),
    });
    
    // =========================================================================
    // Generation_MP3_MP3 - MP3→MP3 2nd gen (should FAIL) - 2 files
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("Generation_MP3_MP3/Instant_Destiny_24sample_mp3_gen2.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Instant Destiny - MP3→MP3 (2nd gen)".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "Generation_MP3_MP3".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("Generation_MP3_MP3/Paranoid_Android__Remastered__16sample_mp3_gen2.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Paranoid Android - MP3→MP3 (2nd gen)".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "Generation_MP3_MP3".to_string(),
    });
    
    // =========================================================================
    // Generation_Opus_MP3 - Opus→MP3 cross-codec (should FAIL) - 1 file
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("Generation_Opus_MP3/Instant_Destiny_24sample_opus_to_mp3.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Instant Destiny - Opus→MP3".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "Generation_Opus_MP3".to_string(),
    });
    
    // =========================================================================
    // MP3_128_Boundary - MP3 128kbps (should FAIL) - 4 files
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("MP3_128_Boundary/An_Ending__Ascent___Remastered_2019__16sample_mp3_128k.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "An Ending (Ascent) - MP3 128k".to_string(),
        genre: "AmbientDrone".to_string(),
        defect_category: "MP3_128_Boundary".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("MP3_128_Boundary/Could_You_Be_Loved__Album_Version__16sample_mp3_128k.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Could You Be Loved - MP3 128k".to_string(),
        genre: "ReggaeDub".to_string(),
        defect_category: "MP3_128_Boundary".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("MP3_128_Boundary/Jelmore_24sample_mp3_128k.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Jelmore - MP3 128k".to_string(),
        genre: "Folk".to_string(),
        defect_category: "MP3_128_Boundary".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("MP3_128_Boundary/Open_Your_Heart_16sample_mp3_128k.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Open Your Heart - MP3 128k".to_string(),
        genre: "Pop".to_string(),
        defect_category: "MP3_128_Boundary".to_string(),
    });
    
    // =========================================================================
    // MP3_192_Mid - MP3 192kbps (should FAIL) - 1 file
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("MP3_192_Mid/Boogieman_24sample_mp3_192k.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Boogieman - MP3 192k".to_string(),
        genre: "HipHopRnB".to_string(),
        defect_category: "MP3_192_Mid".to_string(),
    });
    
    // =========================================================================
    // MP3_320_HighQuality - MP3 320kbps (should FAIL) - 3 files
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("MP3_320_HighQuality/An_Ending__Ascent___Remastered_2019__16sample_mp3_320k.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "An Ending (Ascent) - MP3 320k".to_string(),
        genre: "AmbientDrone".to_string(),
        defect_category: "MP3_320_HighQuality".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("MP3_320_HighQuality/Melatonin_24sample_mp3_320k.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Melatonin - MP3 320k".to_string(),
        genre: "Rock".to_string(),
        defect_category: "MP3_320_HighQuality".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("MP3_320_HighQuality/Missa_Pange_Lingua_-_Kyrie_24sample_mp3_320k.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Missa Pange Lingua - MP3 320k".to_string(),
        genre: "Classical".to_string(),
        defect_category: "MP3_320_HighQuality".to_string(),
    });
    
    // =========================================================================
    // MP3_V0_VBR - MP3 VBR V0 (should FAIL) - 2 files
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("MP3_V0_VBR/Different_Masks_For_Different_Days__Live_from_Echostage__Washington__24sample_mp3_v0.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Different Masks - MP3 VBR V0".to_string(),
        genre: "ElectronicDance".to_string(),
        defect_category: "MP3_V0_VBR".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("MP3_V0_VBR/Instant_Destiny_24sample_mp3_v0.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Instant Destiny - MP3 VBR V0".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "MP3_V0_VBR".to_string(),
    });
    
    // =========================================================================
    // MP3_V4_VBR - MP3 VBR V4 (should FAIL) - 2 files
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("MP3_V4_VBR/Different_Masks_For_Different_Days__Live_from_Echostage__Washington__24sample_mp3_v4.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Different Masks - MP3 VBR V4".to_string(),
        genre: "ElectronicDance".to_string(),
        defect_category: "MP3_V4_VBR".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("MP3_V4_VBR/Instant_Destiny_24sample_mp3_v4.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Instant Destiny - MP3 VBR V4".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "MP3_V4_VBR".to_string(),
    });
    
    // =========================================================================
    // Opus_128_Mid - Opus 128kbps (should FAIL) - 2 files
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("Opus_128_Mid/Boogieman_24sample_opus_128k.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OpusTranscode".to_string()],
        description: "Boogieman - Opus 128k".to_string(),
        genre: "HipHopRnB".to_string(),
        defect_category: "Opus_128_Mid".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("Opus_128_Mid/Different_Masks_For_Different_Days__Live_from_Echostage__Washington__24sample_opus_128k.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OpusTranscode".to_string()],
        description: "Different Masks - Opus 128k".to_string(),
        genre: "ElectronicDance".to_string(),
        defect_category: "Opus_128_Mid".to_string(),
    });
    
    // =========================================================================
    // Opus_192_High - Opus 192kbps (should FAIL) - 1 file
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("Opus_192_High/Alright_16sample_opus_192k.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OpusTranscode".to_string()],
        description: "Alright - Opus 192k".to_string(),
        genre: "SoulFunk".to_string(),
        defect_category: "Opus_192_High".to_string(),
    });
    
    // =========================================================================
    // Opus_64_Low - Opus 64kbps (should FAIL) - 1 file
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("Opus_64_Low/Missa_Pange_Lingua_-_Kyrie_24sample_opus_64k.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OpusTranscode".to_string()],
        description: "Missa Pange Lingua - Opus 64k".to_string(),
        genre: "Classical".to_string(),
        defect_category: "Opus_64_Low".to_string(),
    });
    
    // =========================================================================
    // SampleRate_44to96 - 44.1→96kHz upsampling (should FAIL) - 2 files
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("SampleRate_44to96/Different_Masks_For_Different_Days__Live_from_Echostage__Washington__24sample_44to96_upsampled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Upsampled".to_string()],
        description: "Different Masks - 44.1→96kHz upsampled".to_string(),
        genre: "ElectronicDance".to_string(),
        defect_category: "SampleRate_44to96".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("SampleRate_44to96/inconsist_24sample_44to96_upsampled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Upsampled".to_string()],
        description: "inconsist - 44.1→96kHz upsampled".to_string(),
        genre: "AmbientDrone".to_string(),
        defect_category: "SampleRate_44to96".to_string(),
    });
    
    // =========================================================================
    // SampleRate_48to96 - 48→96kHz upsampling (should FAIL) - 1 file
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("SampleRate_48to96/inconsist_24sample_48to96_upsampled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Upsampled".to_string()],
        description: "inconsist - 48→96kHz upsampled".to_string(),
        genre: "AmbientDrone".to_string(),
        defect_category: "SampleRate_48to96".to_string(),
    });
    
    // =========================================================================
    // Vorbis_Q3_Low - Vorbis Q3 low quality (should FAIL) - 2 files
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("Vorbis_Q3_Low/Instant_Destiny_24sample_vorbis_q3.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OggVorbisTranscode".to_string()],
        description: "Instant Destiny - Vorbis Q3".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "Vorbis_Q3_Low".to_string(),
    });
    
    cases.push(GenreTestCase {
        file_path: base.join("Vorbis_Q3_Low/Paranoid_Android__Remastered__16sample_vorbis_q3.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OggVorbisTranscode".to_string()],
        description: "Paranoid Android - Vorbis Q3".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "Vorbis_Q3_Low".to_string(),
    });
    
    // =========================================================================
    // Vorbis_Q7_High - Vorbis Q7 high quality (should FAIL) - 1 file
    // =========================================================================
    
    cases.push(GenreTestCase {
        file_path: base.join("Vorbis_Q7_High/Instant_Destiny_24sample_vorbis_q7.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OggVorbisTranscode".to_string()],
        description: "Instant Destiny - Vorbis Q7".to_string(),
        genre: "Alternative".to_string(),
        defect_category: "Vorbis_Q7_High".to_string(),
    });
    
    // Total: 45 test cases covering all categories from manifest.txt
    
    cases
}
