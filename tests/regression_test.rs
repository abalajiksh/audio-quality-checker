// tests/regression_test.rs
// REGRESSION Test Suite - Comprehensive ground truth validation
// Uses full TestFiles/ from MinIO for weekly CI runs
//
// This test suite covers:
// - All original master files
// - All sample rate conversions
// - All bit depth conversions
// - All lossy codec transcodes at various bitrates
// - Complex transcoding chains (MasterScript)
// - Generational loss scenarios

use std::process::Command;
use std::path::{Path, PathBuf};

struct TestCase {
    file_path: String,
    should_pass: bool,
    category: String,
    description: String,
}

struct TestResult {
    passed: bool,
    expected: bool,
    defects_found: Vec<String>,
    quality_score: f32,
}

/// Main regression test - runs against full TestFiles
#[test]
fn test_regression_suite() {
    let binary_path = get_binary_path();
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let test_base = project_root.join("TestFiles");

    assert!(
        test_base.exists(),
        "TestFiles directory not found at: {}. \
         Download TestFiles.zip from MinIO for regression tests.",
        test_base.display()
    );

    println!("\n{}", "=".repeat(70));
    println!("REGRESSION TEST SUITE - Full Ground Truth Validation");
    println!("Using: {}", test_base.display());
    println!("{}\n", "=".repeat(70));

    let test_cases = define_regression_tests(&test_base);
    
    let mut passed = 0;
    let mut failed = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;
    let mut category_results: std::collections::HashMap<String, (u32, u32)> = std::collections::HashMap::new();

    println!("Running {} regression tests...\n", test_cases.len());

    for (idx, test_case) in test_cases.iter().enumerate() {
        // Skip if file doesn't exist (some MasterScript files may not be generated)
        if !Path::new(&test_case.file_path).exists() {
            println!("[{:3}/{}] SKIP: {} (file not found)", idx + 1, test_cases.len(), test_case.description);
            continue;
        }

        let result = run_test(&binary_path, test_case);
        let entry = category_results.entry(test_case.category.clone()).or_insert((0, 0));

        if result.passed == result.expected {
            passed += 1;
            entry.0 += 1;
            if idx % 10 == 0 {
                println!("[{:3}/{}] ✓ {}", idx + 1, test_cases.len(), test_case.description);
            }
        } else {
            failed += 1;
            entry.1 += 1;

            if result.passed && !result.expected {
                false_negatives += 1;
                println!("[{:3}/{}] ✗ FALSE NEG: {}", idx + 1, test_cases.len(), test_case.description);
                println!("         Expected defects, got CLEAN (score: {:.1}%)", result.quality_score * 100.0);
            } else {
                false_positives += 1;
                println!("[{:3}/{}] ✗ FALSE POS: {}", idx + 1, test_cases.len(), test_case.description);
                println!("         Expected CLEAN, detected: {:?}", result.defects_found);
            }
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("REGRESSION RESULTS BY CATEGORY");
    println!("{}", "=".repeat(70));
    
    for (category, (pass, fail)) in &category_results {
        let total = pass + fail;
        let pct = if total > 0 { (*pass as f32 / total as f32) * 100.0 } else { 0.0 };
        println!("  {:<30} {:>3}/{:<3} ({:.1}%)", category, pass, total, pct);
    }

    println!("\n{}", "=".repeat(70));
    println!("OVERALL REGRESSION RESULTS");
    println!("{}", "=".repeat(70));
    let total = passed + failed;
    println!("Total Tests:     {}", total);
    println!("Passed:          {} ({:.1}%)", passed, (passed as f32 / total as f32) * 100.0);
    println!("Failed:          {}", failed);
    println!("  False Positives: {} (clean files marked as defective)", false_positives);
    println!("  False Negatives: {} (defective files marked as clean)", false_negatives);
    println!("{}", "=".repeat(70));

    assert_eq!(failed, 0, "Regression failed: {} test(s) did not pass", failed);
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

    #[cfg(not(windows))]
    {
        if release_path.exists() {
            return release_path;
        } else if debug_path.exists() {
            return debug_path;
        }
    }

    panic!("Binary not found. Run: cargo build --release");
}

fn define_regression_tests(base: &Path) -> Vec<TestCase> {
    let mut cases = Vec::new();

    // =========================================================================
    // CLEANORIGIN - Original master files
    // =========================================================================
    
    cases.push(TestCase {
        file_path: base.join("CleanOrigin/input96.flac").to_string_lossy().to_string(),
        should_pass: true,
        category: "CleanOrigin".to_string(),
        description: "CleanOrigin: 96kHz 24-bit master".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("CleanOrigin/input192.flac").to_string_lossy().to_string(),
        should_pass: false,  // 16-bit source
        category: "CleanOrigin".to_string(),
        description: "CleanOrigin: 192kHz (16-bit source)".to_string(),
    });

    // =========================================================================
    // CLEANTRANSCODED - Honest 16-bit transcodes → PASS
    // =========================================================================
    
    cases.push(TestCase {
        file_path: base.join("CleanTranscoded/input96_16bit.flac").to_string_lossy().to_string(),
        should_pass: true,
        category: "CleanTranscoded".to_string(),
        description: "CleanTranscoded: 96kHz 16-bit".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("CleanTranscoded/input192_16bit.flac").to_string_lossy().to_string(),
        should_pass: true,
        category: "CleanTranscoded".to_string(),
        description: "CleanTranscoded: 192kHz 16-bit".to_string(),
    });

    // =========================================================================
    // RESAMPLE96 - From 96kHz source (24-bit)
    // Downsample = PASS, Upsample = FAIL
    // =========================================================================
    
    let resample96_files = [
        ("input96_44.flac", true, "96→44.1kHz (downsample)"),
        ("input96_48.flac", true, "96→48kHz (downsample)"),
        ("input96_88.flac", true, "96→88.2kHz (downsample)"),
        ("input96_176.flac", false, "96→176.4kHz (upsample)"),
        ("input96_192.flac", false, "96→192kHz (upsample)"),
    ];

    for (file, should_pass, desc) in resample96_files {
        cases.push(TestCase {
            file_path: base.join("Resample96").join(file).to_string_lossy().to_string(),
            should_pass,
            category: "Resample96".to_string(),
            description: format!("Resample96: {}", desc),
        });
    }

    // =========================================================================
    // RESAMPLE192 - From 192kHz source (16-bit)
    // ALL FAIL due to 16-bit source inheritance
    // =========================================================================
    
    let resample192_files = [
        "input192_44.flac", "input192_48.flac", "input192_88.flac",
        "input192_96.flac", "input192_176.flac",
    ];

    for file in resample192_files {
        cases.push(TestCase {
            file_path: base.join("Resample192").join(file).to_string_lossy().to_string(),
            should_pass: false,  // 16-bit source
            category: "Resample192".to_string(),
            description: format!("Resample192: {} (16-bit source)", file),
        });
    }

    // =========================================================================
    // UPSCALE16 - 16-bit padded to 24-bit → FAIL
    // =========================================================================
    
    cases.push(TestCase {
        file_path: base.join("Upscale16/output96_16bit.flac").to_string_lossy().to_string(),
        should_pass: false,
        category: "Upscale16".to_string(),
        description: "Upscale16: 96kHz fake 24-bit".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscale16/output192_16bit.flac").to_string_lossy().to_string(),
        should_pass: false,
        category: "Upscale16".to_string(),
        description: "Upscale16: 192kHz fake 24-bit".to_string(),
    });

    // =========================================================================
    // UPSCALED - Lossy codec transcodes → FAIL
    // =========================================================================
    
    let upscaled_files = [
        ("input96_mp3.flac", "96kHz from MP3"),
        ("input192_mp3.flac", "192kHz from MP3"),
        ("input96_m4a.flac", "96kHz from AAC"),
        ("input192_m4a.flac", "192kHz from AAC"),
        ("input96_opus.flac", "96kHz from Opus"),
        ("input192_opus.flac", "192kHz from Opus"),
        ("input96_ogg.flac", "96kHz from Vorbis"),
        ("input192_ogg.flac", "192kHz from Vorbis"),
    ];

    for (file, desc) in upscaled_files {
        cases.push(TestCase {
            file_path: base.join("Upscaled").join(file).to_string_lossy().to_string(),
            should_pass: false,
            category: "Upscaled".to_string(),
            description: format!("Upscaled: {}", desc),
        });
    }

    // =========================================================================
    // MASTERSCRIPT - Complex transcoding chains
    // test96_* = from 24-bit source, test192_* = from 16-bit source
    // =========================================================================
    
    let masterscript = base.join("MasterScript");
    
    // test96 series - from genuine 24-bit source
    // Original reference passes
    cases.push(TestCase {
        file_path: masterscript.join("test96_original.flac").to_string_lossy().to_string(),
        should_pass: true,
        category: "MasterScript-96".to_string(),
        description: "MasterScript: test96 original (reference)".to_string(),
    });

    // Bit depth degradation
    cases.push(TestCase {
        file_path: masterscript.join("test96_16bit_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        category: "MasterScript-96".to_string(),
        description: "MasterScript: test96 16-bit upscaled".to_string(),
    });

    // Sample rate upscaling
    for rate in ["44", "48"] {
        cases.push(TestCase {
            file_path: masterscript.join(format!("test96_resampled_{}_upscaled.flac", rate)).to_string_lossy().to_string(),
            should_pass: false,
            category: "MasterScript-96".to_string(),
            description: format!("MasterScript: test96 resampled {} upscaled", rate),
        });
    }

    // MP3 at various bitrates
    for bitrate in ["128k", "192k", "256k", "320k", "v0", "v2", "v4"] {
        cases.push(TestCase {
            file_path: masterscript.join(format!("test96_mp3_{}_upscaled.flac", bitrate)).to_string_lossy().to_string(),
            should_pass: false,
            category: "MasterScript-96-MP3".to_string(),
            description: format!("MasterScript: test96 MP3 {}", bitrate),
        });
    }

    // AAC at various bitrates
    for bitrate in ["128k", "192k", "256k", "320k"] {
        cases.push(TestCase {
            file_path: masterscript.join(format!("test96_aac_{}_upscaled.flac", bitrate)).to_string_lossy().to_string(),
            should_pass: false,
            category: "MasterScript-96-AAC".to_string(),
            description: format!("MasterScript: test96 AAC {}", bitrate),
        });
    }

    // Opus at various bitrates
    for bitrate in ["64k", "96k", "128k", "192k"] {
        cases.push(TestCase {
            file_path: masterscript.join(format!("test96_opus_{}_upscaled.flac", bitrate)).to_string_lossy().to_string(),
            should_pass: false,
            category: "MasterScript-96-Opus".to_string(),
            description: format!("MasterScript: test96 Opus {}", bitrate),
        });
    }

    // Vorbis at various qualities
    for quality in ["q3", "q5", "q7", "q9"] {
        cases.push(TestCase {
            file_path: masterscript.join(format!("test96_vorbis_{}_upscaled.flac", quality)).to_string_lossy().to_string(),
            should_pass: false,
            category: "MasterScript-96-Vorbis".to_string(),
            description: format!("MasterScript: test96 Vorbis {}", quality),
        });
    }

    // Combined degradations
    cases.push(TestCase {
        file_path: masterscript.join("test96_16bit_44_mp3_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        category: "MasterScript-96-Combined".to_string(),
        description: "MasterScript: test96 16-bit+44kHz+MP3".to_string(),
    });

    // Generational loss
    let gen_loss = [
        "test96_mp3_to_mp3_upscaled.flac",
        "test96_mp3_to_aac_upscaled.flac",
        "test96_opus_to_mp3_upscaled.flac",
    ];
    for file in gen_loss {
        cases.push(TestCase {
            file_path: masterscript.join(file).to_string_lossy().to_string(),
            should_pass: false,
            category: "MasterScript-96-GenLoss".to_string(),
            description: format!("MasterScript: {}", file.replace("_upscaled.flac", "")),
        });
    }

    // test192 series - ALL FAIL due to 16-bit source
    cases.push(TestCase {
        file_path: masterscript.join("test192_original.flac").to_string_lossy().to_string(),
        should_pass: false,  // 16-bit source
        category: "MasterScript-192".to_string(),
        description: "MasterScript: test192 original (16-bit source)".to_string(),
    });

    // MP3 from 192 source
    for bitrate in ["128k", "192k", "256k", "320k", "v0", "v2", "v4"] {
        cases.push(TestCase {
            file_path: masterscript.join(format!("test192_mp3_{}_upscaled.flac", bitrate)).to_string_lossy().to_string(),
            should_pass: false,
            category: "MasterScript-192-MP3".to_string(),
            description: format!("MasterScript: test192 MP3 {} (16-bit+MP3)", bitrate),
        });
    }

    // AAC from 192 source
    for bitrate in ["128k", "192k", "256k", "320k"] {
        cases.push(TestCase {
            file_path: masterscript.join(format!("test192_aac_{}_upscaled.flac", bitrate)).to_string_lossy().to_string(),
            should_pass: false,
            category: "MasterScript-192-AAC".to_string(),
            description: format!("MasterScript: test192 AAC {} (16-bit+AAC)", bitrate),
        });
    }

    // Opus from 192 source
    for bitrate in ["64k", "96k", "128k", "192k"] {
        cases.push(TestCase {
            file_path: masterscript.join(format!("test192_opus_{}_upscaled.flac", bitrate)).to_string_lossy().to_string(),
            should_pass: false,
            category: "MasterScript-192-Opus".to_string(),
            description: format!("MasterScript: test192 Opus {} (16-bit+Opus)", bitrate),
        });
    }

    // Vorbis from 192 source
    for quality in ["q3", "q5", "q7", "q9"] {
        cases.push(TestCase {
            file_path: masterscript.join(format!("test192_vorbis_{}_upscaled.flac", quality)).to_string_lossy().to_string(),
            should_pass: false,
            category: "MasterScript-192-Vorbis".to_string(),
            description: format!("MasterScript: test192 Vorbis {} (16-bit+Vorbis)", quality),
        });
    }

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

    // Parse output
    let has_issues = stdout.contains("ISSUES DETECTED") || stdout.contains("✗");
    let is_clean = stdout.contains("CLEAN") && !has_issues;
    let is_lossless = stdout.contains("likely lossless");

    // Extract quality score
    let quality_score = extract_quality_score(&stdout);

    let mut defects_found = Vec::new();

    if stdout.contains("MP3") || stdout.contains("Mp3") {
        defects_found.push("Mp3".to_string());
    }
    if stdout.contains("AAC") || stdout.contains("Aac") {
        defects_found.push("AAC".to_string());
    }
    if stdout.contains("Opus") {
        defects_found.push("Opus".to_string());
    }
    if stdout.contains("Vorbis") || stdout.contains("Ogg") {
        defects_found.push("Vorbis".to_string());
    }
    if stdout.contains("Bit depth") || stdout.contains("bit depth") {
        defects_found.push("BitDepth".to_string());
    }
    if stdout.contains("Upsampled") || stdout.contains("upsampled") {
        defects_found.push("Upsampled".to_string());
    }
    if stdout.contains("Spectral") {
        defects_found.push("Spectral".to_string());
    }

    TestResult {
        passed: is_clean || is_lossless,
        expected: test_case.should_pass,
        defects_found,
        quality_score,
    }
}

fn extract_quality_score(output: &str) -> f32 {
    if let Some(pos) = output.find("Quality Score:") {
        let after = &output[pos + 14..];
        if let Some(end) = after.find('%') {
            let score_str = after[..end].trim();
            if let Ok(score) = score_str.parse::<f32>() {
                return score / 100.0;
            }
        }
    }
    0.0
}

#[test]
fn test_binary_exists() {
    let binary_path = get_binary_path();
    assert!(binary_path.exists(), "Binary not found at {:?}", binary_path);
}
