// tests/qualification_genre_test.rs

// QUALIFICATION Genre Test Suite - Compact subset for CI/CD quick validation
// Uses GenreTestSuiteLite (~50 files) for fast validation on every push
//
// Purpose: Quick sanity check with real music samples
// - Tests representative samples from Control and major defect categories
// - Focuses on high-confidence detection scenarios
// - Parallel execution (4 threads) for faster CI/CD

use std::env;
use std::fs;
use std::process::Command;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Clone)]
struct GenreTestCase {
    file_path: String,
    should_pass: bool,
    #[allow(dead_code)]
    expected_defects: Vec<String>,
    description: String,
    #[allow(dead_code)]
    genre: String,
    defect_category: String,
}

#[derive(Debug)]
struct TestResult {
    passed: bool,
    expected: bool,
    defects_found: Vec<String>,
    description: String,
    category: String,
    #[allow(dead_code)]
    file: String,
}

#[test]
fn test_qualification_genre_suite() {
    let binary_path = get_binary_path();
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    
    // Try GenreTestSuiteLite first, then TestSuite as fallback
    let test_base = if project_root.join("GenreTestSuiteLite").exists() {
        project_root.join("GenreTestSuiteLite")
    } else if project_root.join("TestSuite").exists() {
        project_root.join("TestSuite")
    } else {
        panic!(
            "Neither GenreTestSuiteLite nor TestSuite directory found. \
             Download GenreTestSuiteLite.zip from MinIO for qualification genre tests."
        );
    };

    println!("\n{}", "=".repeat(70));
    println!("QUALIFICATION GENRE TEST SUITE (Parallel Execution)");
    println!("Using: {}", test_base.display());
    println!("{}\n", "=".repeat(70));

    // Scan categories and build test cases
    let test_cases = scan_and_build_test_cases(&test_base);
    let total_tests = test_cases.len();

    if total_tests == 0 {
        println!("WARNING: No test files found in {}", test_base.display());
        println!("Expected FLAC files in category subdirectories or flat structure.");
        return;
    }

    println!("Found {} files across {} categories\n", total_tests, count_categories(&test_cases));

    // Run tests in parallel with 4 threads
    let results = run_tests_parallel(&binary_path, test_cases, 4);

    // Analyze results by category
    let mut passed = 0;
    let mut failed = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;
    let mut results_by_category: std::collections::HashMap<String, Vec<&TestResult>> =
        std::collections::HashMap::new();

    for result in &results {
        results_by_category
            .entry(result.category.clone())
            .or_insert_with(Vec::new)
            .push(result);

        if result.passed == result.expected {
            passed += 1;
        } else {
            failed += 1;
            if result.passed && !result.expected {
                false_negatives += 1;
                println!(
                    "✗ FALSE NEGATIVE [{}]: {}",
                    result.category, result.description
                );
            } else {
                false_positives += 1;
                println!(
                    "✗ FALSE POSITIVE [{}]: {} - Found: {:?}",
                    result.category, result.description, result.defects_found
                );
            }
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("QUALIFICATION GENRE RESULTS");
    println!("{}", "=".repeat(70));
    println!("Total Tests: {}", total_tests);
    println!(
        "Passed: {} ({:.1}%)",
        passed,
        (passed as f32 / total_tests as f32) * 100.0
    );
    println!("Failed: {}", failed);
    println!("  False Positives: {}", false_positives);
    println!("  False Negatives: {}", false_negatives);

    // Category breakdown
    println!("\n{}", "-".repeat(70));
    println!("Results by Category:");
    println!("{}", "-".repeat(70));
    
    let mut categories: Vec<_> = results_by_category.iter().collect();
    categories.sort_by_key(|(k, _)| k.as_str());
    
    for (category, cat_results) in categories {
        let cat_passed = cat_results.iter().filter(|r| r.passed == r.expected).count();
        let cat_total = cat_results.len();
        println!(
            "{:35} {:3}/{:3} ({:.0}%)",
            category,
            cat_passed,
            cat_total,
            (cat_passed as f32 / cat_total as f32) * 100.0
        );
    }
    println!("{}", "=".repeat(70));

    assert_eq!(
        failed, 0,
        "Qualification genre tests failed: {} test(s) did not pass",
        failed
    );
}

fn scan_and_build_test_cases(base: &Path) -> Vec<GenreTestCase> {
    let mut cases = Vec::new();

    // Read entries in the base directory
    let entries = match fs::read_dir(base) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!("Failed to read test directory: {}", e);
            return cases;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();
        
        // Check if it's a directory (subdirectory structure)
        if path.is_dir() {
            let category = match path.file_name().and_then(|n| n.to_str()) {
                Some(name) => name.to_string(),
                None => continue,
            };

            // Scan FLAC files in this category subdirectory
            let files = match fs::read_dir(&path) {
                Ok(files) => files,
                Err(_) => continue,
            };

            for file_entry in files {
                let file_entry = match file_entry {
                    Ok(f) => f,
                    Err(_) => continue,
                };

                let file_path = file_entry.path();
                if file_path.extension().and_then(|e| e.to_str()) != Some("flac") {
                    continue;
                }

                let filename = match file_path.file_name().and_then(|n| n.to_str()) {
                    Some(name) => name.to_string(),
                    None => continue,
                };

                // Determine expected result based on category
                let (should_pass, expected_defects) = categorize_expected_result(&category);
                let genre_info = extract_genre_from_filename(&filename);

                cases.push(GenreTestCase {
                    file_path: file_path.to_string_lossy().to_string(),
                    should_pass,
                    expected_defects,
                    description: filename.clone(),
                    genre: genre_info,
                    defect_category: category.clone(),
                });
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("flac") {
            // Handle flat file structure - extract category from filename prefix
            let filename = match path.file_name().and_then(|n| n.to_str()) {
                Some(name) => name.to_string(),
                None => continue,
            };

            // Extract category from filename pattern: "CategoryName__description.flac"
            // Examples:
            //   "AAC_128_Low__Different_Masks__24sample_aac_128k.flac" -> "AAC_128_Low"
            //   "BitDepth_16to24__Boogieman_24sample_16to24_upscaled.flac" -> "BitDepth_16to24"
            let category = if let Some(pos) = filename.find("__") {
                filename[..pos].to_string()
            } else {
                // If no "__" delimiter, try to extract from first underscore pattern
                filename
                    .split('_')
                    .take(3) // Take first 3 parts (e.g., "AAC", "128", "Low")
                    .collect::<Vec<_>>()
                    .join("_")
            };

            // Determine expected result based on category
            let (should_pass, expected_defects) = categorize_expected_result(&category);
            let genre_info = extract_genre_from_filename(&filename);

            cases.push(GenreTestCase {
                file_path: path.to_string_lossy().to_string(),
                should_pass,
                expected_defects,
                description: filename.clone(),
                genre: genre_info,
                defect_category: category.clone(),
            });
        }
    }

    // Sort by category then filename for consistent ordering
    cases.sort_by(|a, b| {
        a.defect_category
            .cmp(&b.defect_category)
            .then(a.description.cmp(&b.description))
    });

    cases
}

fn categorize_expected_result(category: &str) -> (bool, Vec<String>) {
    match category {
        "Control_Original" => (true, vec![]),

        cat if cat.starts_with("MP3_") || cat.contains("MP3") => {
            (false, vec!["Mp3Transcode".to_string()])
        }

        cat if cat.starts_with("AAC_") => (false, vec!["AacTranscode".to_string()]),

        cat if cat.starts_with("Opus_") || cat.contains("Opus") => {
            (false, vec!["OpusTranscode".to_string()])
        }

        cat if cat.starts_with("Vorbis_") => (false, vec!["OggVorbisTranscode".to_string()]),

        "BitDepth_16to24" => (false, vec!["BitDepthMismatch".to_string()]),

        "Combined_16bit_44khz" => (
            false,
            vec!["BitDepthMismatch".to_string(), "Upsampled".to_string()],
        ),

        "Combined_MP3_128_From_CD" => (
            false,
            vec!["Mp3Transcode".to_string(), "BitDepthMismatch".to_string()],
        ),

        cat if cat.starts_with("SampleRate_") => (false, vec!["Upsampled".to_string()]),

        cat if cat.starts_with("Edge_") && cat.contains("Resample") => {
            (false, vec!["Upsampled".to_string()])
        }

        cat if cat.starts_with("Generation_") => {
            // Multi-generation transcodes
            if cat.contains("MP3") {
                (false, vec!["Mp3Transcode".to_string()])
            } else if cat.contains("AAC") {
                (false, vec!["AacTranscode".to_string()])
            } else if cat.contains("Opus") {
                (false, vec!["OpusTranscode".to_string()])
            } else {
                (false, vec![])
            }
        }

        _ => (false, vec![]),
    }
}

fn extract_genre_from_filename(filename: &str) -> String {
    // Extract genre information from filename patterns
    if filename.contains("Boogieman") {
        "HipHopRnB".to_string()
    } else if filename.contains("Paranoid_Android") || filename.contains("Instant_Destiny") {
        "Alternative".to_string()
    } else if filename.contains("inconsist") || filename.contains("An_Ending") {
        "AmbientDrone".to_string()
    } else if filename.contains("Different_Masks") || filename.contains("Windowlicker") {
        "ElectronicDance".to_string()
    } else if filename.contains("Could_You_Be_Loved") {
        "ReggaeDub".to_string()
    } else if filename.contains("MALAMENTE") || filename.contains("Chan_Chan") {
        "LatinWorld".to_string()
    } else if filename.contains("Wake_Up") || filename.contains("Punisher") {
        "Indie".to_string()
    } else if filename.contains("Pride_and_Joy") {
        "Blues".to_string()
    } else if filename.contains("Brandenburg") || filename.contains("Missa_Pange") {
        "Classical".to_string()
    } else if filename.contains("Dream_of_Arrakis") || filename.contains("Bene_Gesserit") {
        "SoundtrackScore".to_string()
    } else if filename.contains("Enter_Sandman") || filename.contains("Crack_the_Skye") {
        "Metal".to_string()
    } else if filename.contains("So_What") {
        "Jazz".to_string()
    } else {
        "Unknown".to_string()
    }
}

fn count_categories(cases: &[GenreTestCase]) -> usize {
    let mut categories: std::collections::HashSet<String> = std::collections::HashSet::new();
    for case in cases {
        categories.insert(case.defect_category.clone());
    }
    categories.len()
}

fn run_tests_parallel(
    binary: &Path,
    test_cases: Vec<GenreTestCase>,
    num_threads: usize,
) -> Vec<TestResult> {
    let binary = binary.to_path_buf();
    let test_cases = Arc::new(test_cases);
    let results = Arc::new(Mutex::new(Vec::new()));
    let index = Arc::new(Mutex::new(0usize));
    let mut handles = Vec::new();

    println!("Running tests with {} parallel threads...\n", num_threads);

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

                // Progress indicator every 10 files
                if current_idx > 0 && current_idx % 10 == 0 {
                    println!("Progress: {}/{} tests completed", current_idx, test_cases.len());
                }

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

/// Parse defects from audiocheckr output
fn parse_defects_from_output(stdout: &str) -> Vec<String> {
    let mut defects_found = Vec::new();
    let stdout_lower = stdout.to_lowercase();

    // Look for specific defect patterns in the output
    // Check for transcode detections
    if (stdout_lower.contains("mp3") && stdout_lower.contains("transcode"))
        || stdout_lower.contains("mp3transcode")
    {
        defects_found.push("Mp3Transcode".to_string());
    }
    if (stdout_lower.contains("aac") && stdout_lower.contains("transcode"))
        || stdout_lower.contains("aactranscode")
    {
        defects_found.push("AacTranscode".to_string());
    }
    if (stdout_lower.contains("opus") && stdout_lower.contains("transcode"))
        || stdout_lower.contains("opustranscode")
    {
        defects_found.push("OpusTranscode".to_string());
    }
    if ((stdout_lower.contains("vorbis") || stdout_lower.contains("ogg"))
        && stdout_lower.contains("transcode"))
        || stdout_lower.contains("oggvorbistranscode")
    {
        defects_found.push("OggVorbisTranscode".to_string());
    }

    // Check for bit depth issues
    if stdout_lower.contains("bit depth mismatch")
        || stdout_lower.contains("bitdepthmismatch")
        || (stdout_lower.contains("bit depth") && stdout_lower.contains("mismatch"))
    {
        defects_found.push("BitDepthMismatch".to_string());
    }

    // Check for upsampling
    if stdout_lower.contains("upsampled")
        || (stdout_lower.contains("upsample") && !stdout_lower.contains("not upsampled"))
    {
        defects_found.push("Upsampled".to_string());
    }

    // Check for spectral artifacts
    if stdout_lower.contains("spectral artifact") {
        defects_found.push("SpectralArtifacts".to_string());
    }

    defects_found
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

    // First, parse all defects from the output
    let defects_found = parse_defects_from_output(&stdout);

    // Check for explicit status indicators in output
    let has_explicit_issues = stdout.contains("ISSUES DETECTED")
        || stdout.contains("✗ ISSUES")
        || stdout.to_lowercase().contains("issues detected");
    let has_explicit_clean = (stdout.contains("CLEAN") || stdout.to_lowercase().contains("clean"))
        && !has_explicit_issues;

    // FIXED LOGIC: A file is "clean" (passed) if:
    // 1. No defects were parsed from the output, AND
    // 2. There's no explicit "ISSUES DETECTED" message
    // OR
    // 3. There's an explicit "CLEAN" status
    let is_clean = has_explicit_clean || (defects_found.is_empty() && !has_explicit_issues);

    TestResult {
        passed: is_clean,
        expected: test_case.should_pass,
        defects_found,
        description: test_case.description.clone(),
        category: test_case.defect_category.clone(),
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

#[test]
fn test_binary_exists() {
    let binary_path = get_binary_path();
    assert!(binary_path.exists(), "Binary not found at {:?}", binary_path);
}
