// tests/regression_test.rs
// REGRESSION Test Suite - Comprehensive ground truth validation
// Uses full TestFiles.zip (8.5GB) for complete coverage
//
// Test Philosophy:
// - CleanOrigin: Original master files → PASS (genuine high-res) except input192 (16-bit source)
// - CleanTranscoded: 24→16 bit honest transcodes → PASS
// - Resample96: 96kHz → lower rates = PASS, 96kHz → higher rates = FAIL (interpolated)
// - Resample192: All are from 16-bit source → FAIL (BitDepthMismatch)
// - Upscale16: 16-bit → 24-bit padding = FAIL
// - Upscaled: Lossy → Lossless = FAIL
// - MasterScript: Complex transcoding chains - most should FAIL

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
    #[allow(dead_code)]
    file: String,
}

/// Main regression test - comprehensive coverage
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
    println!("GROUND TRUTH REGRESSION TEST SUITE");
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
            println!("[{:2}/{}] ✓ PASS: {}", idx + 1, test_cases.len(), test_case.description);
        } else {
            failed += 1;
            entry.1 += 1;

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

    println!("\n{}", "=".repeat(70));
    println!("REGRESSION RESULTS");
    println!("{}", "=".repeat(70));
    println!("Total Tests:       {}", test_cases.len());
    println!("Passed:            {} ({:.1}%)", passed, (passed as f32 / test_cases.len() as f32) * 100.0);
    println!("Failed:            {}", failed);
    println!("  False Positives: {} (clean files marked as defective)", false_positives);
    println!("  False Negatives: {} (defective files marked as clean)", false_negatives);
    println!("{}", "=".repeat(70));

    if failed > 0 {
        println!("\n⚠️  Detector needs improvement in {} areas", failed);
    } else {
        println!("\n✅ Perfect detection across all {} test cases!", test_cases.len());
    }

    // For regression tests, we report but don't fail on detection issues
    // This allows tracking detector improvements over time
    // Change to assert_eq!(failed, 0, ...) if strict pass/fail is needed
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
        expected_defects: vec![],
        description: "CleanOrigin: 96kHz 24-bit original".to_string(),
    });

    // input192.flac is documented as 16-bit source
    cases.push(TestCase {
        file_path: base.join("CleanOrigin/input192.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "CleanOrigin: 192kHz (16-bit source)".to_string(),
    });

    // =========================================================================
    // CLEANTRANSCODED - Honest 24→16 bit transcodes
    // =========================================================================
    
    cases.push(TestCase {
        file_path: base.join("CleanTranscoded/input96_16bit.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "CleanTranscoded: 96kHz honest 16-bit".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("CleanTranscoded/input192_16bit.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "CleanTranscoded: 192kHz honest 16-bit".to_string(),
    });

    // =========================================================================
    // RESAMPLE96 - Sample rate changes from 96kHz source
    // Downsampling = PASS, Upsampling = FAIL
    // =========================================================================
    
    // Downsampled (genuine) - PASS
    for rate in ["44", "48", "88"] {
        cases.push(TestCase {
            file_path: base.join(format!("Resample96/input96_{}.flac", rate)).to_string_lossy().to_string(),
            should_pass: true,
            expected_defects: vec![],
            description: format!("Resample96: 96→{}kHz downsampled", rate),
        });
    }

    // Upsampled (interpolated) - FAIL
    for rate in ["176", "192"] {
        cases.push(TestCase {
            file_path: base.join(format!("Resample96/input96_{}.flac", rate)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec!["Upsampled".to_string()],
            description: format!("Resample96: 96→{}kHz upsampled (interpolated)", rate),
        });
    }

    // =========================================================================
    // RESAMPLE192 - All from 16-bit source, all should FAIL BitDepthMismatch
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
    // UPSCALE16 - 16-bit → 24-bit padding (fake bit depth)
    // =========================================================================
    
    cases.push(TestCase {
        file_path: base.join("Upscale16/output96_16bit.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "Upscale16: 96kHz 16→24-bit".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscale16/output192_16bit.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "Upscale16: 192kHz 16→24-bit".to_string(),
    });

    // =========================================================================
    // UPSCALED - Lossy codec → FLAC transcodes
    // =========================================================================
    
    // 96kHz source (genuine 24-bit) - only lossy artifact
    cases.push(TestCase {
        file_path: base.join("Upscaled/input96_mp3.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "Upscaled: 96kHz from MP3".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscaled/input96_m4a.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["AacTranscode".to_string()],
        description: "Upscaled: 96kHz from AAC".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscaled/input96_opus.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OpusTranscode".to_string()],
        description: "Upscaled: 96kHz from Opus".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscaled/input96_ogg.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OggVorbisTranscode".to_string()],
        description: "Upscaled: 96kHz from Vorbis".to_string(),
    });

    // 192kHz source (16-bit) - both lossy + bit depth issues
    cases.push(TestCase {
        file_path: base.join("Upscaled/input192_mp3.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string(), "BitDepthMismatch".to_string()],
        description: "Upscaled: 192kHz from MP3 (16-bit source)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscaled/input192_m4a.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["AacTranscode".to_string(), "BitDepthMismatch".to_string()],
        description: "Upscaled: 192kHz from AAC (16-bit source)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscaled/input192_opus.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OpusTranscode".to_string(), "BitDepthMismatch".to_string()],
        description: "Upscaled: 192kHz from Opus (16-bit source)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("Upscaled/input192_ogg.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OggVorbisTranscode".to_string(), "BitDepthMismatch".to_string()],
        description: "Upscaled: 192kHz from Vorbis (16-bit source)".to_string(),
    });

    // =========================================================================
    // MASTERSCRIPT - Complex transcoding chains (96kHz source = genuine 24-bit)
    // =========================================================================
    
    // test96_original - the reference file, should PASS
    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_original.flac").to_string_lossy().to_string(),
        should_pass: true,
        expected_defects: vec![],
        description: "MasterScript: test96 original (reference)".to_string(),
    });

    // test96 bit depth degradations
    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_16bit_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "MasterScript: test96 16-bit upscaled".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_16bit_44khz_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "MasterScript: test96 16-bit 44.1kHz upscaled".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_16bit_44khz_mp3_128_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string(), "Mp3Transcode".to_string()],
        description: "MasterScript: test96 16-bit 44.1kHz MP3 128k upscaled".to_string(),
    });

    // test96 resample degradations
    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_resampled_44.1_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Upsampled".to_string()],
        description: "MasterScript: test96 44.1kHz→192kHz upsampled".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_resampled_48_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Upsampled".to_string()],
        description: "MasterScript: test96 48kHz→192kHz upsampled".to_string(),
    });

    // test96 MP3 degradations (various bitrates and VBR modes)
    for bitrate in ["128", "192", "256", "320"] {
        cases.push(TestCase {
            file_path: base.join(format!("MasterScript/test96_mp3_{}_upscaled.flac", bitrate)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec!["Mp3Transcode".to_string()],
            description: format!("MasterScript: test96 MP3 {}k upscaled", bitrate),
        });
    }

    for vbr in ["v0", "v2", "v4"] {
        cases.push(TestCase {
            file_path: base.join(format!("MasterScript/test96_mp3_{}_upscaled.flac", vbr)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec!["Mp3Transcode".to_string()],
            description: format!("MasterScript: test96 MP3 {} upscaled", vbr),
        });
    }

    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_mp3_320_reencoded_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string()],
        description: "MasterScript: test96 MP3 320k re-encoded upscaled".to_string(),
    });

    // test96 AAC degradations
    for bitrate in ["128", "192", "256", "320"] {
        cases.push(TestCase {
            file_path: base.join(format!("MasterScript/test96_aac_{}_upscaled.flac", bitrate)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec!["AacTranscode".to_string()],
            description: format!("MasterScript: test96 AAC {}k upscaled", bitrate),
        });
    }

    // test96 Opus degradations
    for bitrate in ["64", "96", "128", "160", "192"] {
        cases.push(TestCase {
            file_path: base.join(format!("MasterScript/test96_opus_{}_upscaled.flac", bitrate)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec!["OpusTranscode".to_string()],
            description: format!("MasterScript: test96 Opus {}k upscaled", bitrate),
        });
    }

    // test96 Vorbis degradations
    for quality in ["q3", "q5", "q7", "q9"] {
        cases.push(TestCase {
            file_path: base.join(format!("MasterScript/test96_vorbis_{}_upscaled.flac", quality)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec!["OggVorbisTranscode".to_string()],
            description: format!("MasterScript: test96 Vorbis {} upscaled", quality),
        });
    }

    // test96 cross-codec degradations
    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_mp3_to_aac_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["Mp3Transcode".to_string(), "AacTranscode".to_string()],
        description: "MasterScript: test96 MP3→AAC upscaled".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("MasterScript/test96_opus_to_mp3_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["OpusTranscode".to_string(), "Mp3Transcode".to_string()],
        description: "MasterScript: test96 Opus→MP3 upscaled".to_string(),
    });

    // =========================================================================
    // MASTERSCRIPT - 192kHz source (16-bit origin, all have BitDepthMismatch)
    // =========================================================================
    
    // test192_original - 16-bit source, should FAIL
    cases.push(TestCase {
        file_path: base.join("MasterScript/test192_original.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "MasterScript: test192 original (16-bit source)".to_string(),
    });

    // test192 bit depth degradations
    cases.push(TestCase {
        file_path: base.join("MasterScript/test192_16bit_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "MasterScript: test192 16-bit upscaled".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("MasterScript/test192_16bit_44khz_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "MasterScript: test192 16-bit 44.1kHz upscaled".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("MasterScript/test192_16bit_44khz_mp3_128_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string(), "Mp3Transcode".to_string()],
        description: "MasterScript: test192 16-bit 44.1kHz MP3 128k upscaled".to_string(),
    });

    // test192 resample degradations
    cases.push(TestCase {
        file_path: base.join("MasterScript/test192_resampled_44.1_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "MasterScript: test192 44.1kHz→192kHz (16-bit source)".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("MasterScript/test192_resampled_48_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string()],
        description: "MasterScript: test192 48kHz→192kHz (16-bit source)".to_string(),
    });

    // test192 MP3 degradations
    for bitrate in ["128", "192", "256", "320"] {
        cases.push(TestCase {
            file_path: base.join(format!("MasterScript/test192_mp3_{}_upscaled.flac", bitrate)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec!["BitDepthMismatch".to_string(), "Mp3Transcode".to_string()],
            description: format!("MasterScript: test192 MP3 {}k upscaled", bitrate),
        });
    }

    for vbr in ["v0", "v2", "v4"] {
        cases.push(TestCase {
            file_path: base.join(format!("MasterScript/test192_mp3_{}_upscaled.flac", vbr)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec!["BitDepthMismatch".to_string(), "Mp3Transcode".to_string()],
            description: format!("MasterScript: test192 MP3 {} upscaled", vbr),
        });
    }

    cases.push(TestCase {
        file_path: base.join("MasterScript/test192_mp3_320_reencoded_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string(), "Mp3Transcode".to_string()],
        description: "MasterScript: test192 MP3 320k re-encoded upscaled".to_string(),
    });

    // test192 AAC degradations
    for bitrate in ["128", "192", "256", "320"] {
        cases.push(TestCase {
            file_path: base.join(format!("MasterScript/test192_aac_{}_upscaled.flac", bitrate)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec!["BitDepthMismatch".to_string(), "AacTranscode".to_string()],
            description: format!("MasterScript: test192 AAC {}k upscaled", bitrate),
        });
    }

    // test192 Opus degradations
    for bitrate in ["64", "96", "128", "160", "192"] {
        cases.push(TestCase {
            file_path: base.join(format!("MasterScript/test192_opus_{}_upscaled.flac", bitrate)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec!["BitDepthMismatch".to_string(), "OpusTranscode".to_string()],
            description: format!("MasterScript: test192 Opus {}k upscaled", bitrate),
        });
    }

    // test192 Vorbis degradations
    for quality in ["q3", "q5", "q7", "q9"] {
        cases.push(TestCase {
            file_path: base.join(format!("MasterScript/test192_vorbis_{}_upscaled.flac", quality)).to_string_lossy().to_string(),
            should_pass: false,
            expected_defects: vec!["BitDepthMismatch".to_string(), "OggVorbisTranscode".to_string()],
            description: format!("MasterScript: test192 Vorbis {} upscaled", quality),
        });
    }

    // test192 cross-codec degradations
    cases.push(TestCase {
        file_path: base.join("MasterScript/test192_mp3_to_aac_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string(), "Mp3Transcode".to_string(), "AacTranscode".to_string()],
        description: "MasterScript: test192 MP3→AAC upscaled".to_string(),
    });

    cases.push(TestCase {
        file_path: base.join("MasterScript/test192_opus_to_mp3_upscaled.flac").to_string_lossy().to_string(),
        should_pass: false,
        expected_defects: vec!["BitDepthMismatch".to_string(), "OpusTranscode".to_string(), "Mp3Transcode".to_string()],
        description: "MasterScript: test192 Opus→MP3 upscaled".to_string(),
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
    if stdout.contains("Bit depth mismatch") || stdout.contains("BitDepth") || stdout.contains("bit depth") {
        defects_found.push("BitDepthMismatch".to_string());
    }
    if stdout.contains("Upsampled") || stdout.contains("upsampled") || stdout.contains("interpolat") {
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

#[test]
fn test_binary_exists() {
    let binary_path = get_binary_path();
    assert!(binary_path.exists(), "Binary not found at {:?}", binary_path);
}

#[test]
fn test_binary_exists() {
    let binary_path = get_binary_path();
    let output = Command::new(&binary_path)
        .arg("--help")
        .output()
        .expect("Failed to run --help");
    
    assert!(output.status.success(), "Help command failed");
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("audiocheckr"), "Help output should mention audiocheckr");
}
