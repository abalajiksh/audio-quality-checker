# AudioCheckr

A high-resolution audio quality validator that detects fake lossless files, upsampling, bit-depth padding, and lossy codec transcodes.

## Features

- **Bit Depth Validation**: Detects 16-bit audio padded to 24-bit containers
- **Upsampling Detection**: Identifies files upsampled from lower sample rates
- **Lossy Codec Detection**: Recognizes MP3, AAC, Opus, and Vorbis transcodes
- **Spectral Analysis**: Advanced frequency analysis to identify processing artifacts

## Installation

### From Source

```bash
git clone https://github.com/abalajiksh/audiocheckr.git
cd audiocheckr
cargo build --release
```

Binary will be at `target/release/audiocheckr`

### Usage

```bash
# Basic check
./audiocheckr --input audio.flac

# Check against claimed bit depth
./audiocheckr --input audio.flac --bit-depth 24

# Check for upsampling
./audiocheckr --input audio.flac --check-upsampling

# Full analysis
./audiocheckr --input audio.flac --bit-depth 24 --check-upsampling
```

---

## CI/CD Pipeline

AudioCheckr uses Jenkins for continuous integration with two test modes:

### Test Types

| Type | Trigger | Test Files | Duration | Purpose |
|------|---------|------------|----------|---------|
| **Qualification** | Every push/PR | ~20 files (1.4GB) | ~3 min | Quick validation |
| **Regression** | Weekly scheduled | ~80 files (8.5GB) | ~15 min | Comprehensive ground truth |

### Triggering Builds

#### Automatic Triggers

- **Push to GitHub**: Runs qualification tests
- **Saturday 10:00 AM**: Runs regression tests (scheduled)

#### Manual Triggers (Jenkins UI)

1. Go to Jenkins → AudioCheckr project
2. Click **"Build with Parameters"**
3. Select options:
   - `TEST_TYPE_OVERRIDE`: Choose `REGRESSION` to run full test suite
   - `SKIP_SONARQUBE`: Skip code quality analysis
   - `CLEAN_WORKSPACE_BEFORE`: Fresh start (if seeing issues)
4. Click **Build**

### Test Results

Test results appear in Jenkins UI:
- **Test Result Trend**: Graph showing pass/fail over time
- **Test Results**: Detailed breakdown of each test case
- **Console Output**: Full test output with individual file results

### Pipeline Stages

```
┌─────────────┐   ┌──────────┐   ┌──────────────────┐   ┌─────────┐
│ Pre-flight  │ → │ Checkout │ → │ Download Files   │ → │  Build  │
└─────────────┘   └──────────┘   └──────────────────┘   └─────────┘
                                                              │
                                                              ▼
                              ┌────────────────────────────────────────┐
                              │         Analysis & Tests               │
                              │  ┌──────────────┐  ┌────────────────┐  │
                              │  │  SonarQube   │  │     Tests      │  │
                              │  │  (parallel)  │  │   (parallel)   │  │
                              │  └──────────────┘  └────────────────┘  │
                              └────────────────────────────────────────┘
                                                              │
                                                              ▼
                              ┌────────────────────────────────────────┐
                              │  Cleanup (delete test files & cache)   │
                              └────────────────────────────────────────┘
```

---

## SonarQube Setup

### Quality Gate Webhook (Required for Quality Gate status)

The "Quality Gate" stage times out because SonarQube needs to notify Jenkins when analysis is complete.

**Configure Webhook:**

1. Go to SonarQube → **Project Settings** → **Webhooks**
2. Click **Create**
3. Configure:
   - **Name**: `Jenkins`
   - **URL**: `http://YOUR_JENKINS_URL/sonarqube-webhook/`
   - **Secret**: (leave blank or set if using authentication)
4. Click **Create**

Example URL: `http://192.168.178.100:8080/sonarqube-webhook/`

### Jenkins Configuration

Ensure these are configured in Jenkins:

1. **Manage Jenkins → Configure System → SonarQube servers**
   - Name: `SonarQube-LXC`
   - Server URL: `http://192.168.178.101:9000`
   - Token: (your SonarQube token)

2. **Manage Jenkins → Global Tool Configuration → SonarQube Scanner**
   - Name: `SonarQube-LXC`
   - Install automatically: Yes

---

## Test File Structure

### Directory Layout

```
TestFiles/
├── CleanOrigin/          # Original master recordings
│   ├── input96.flac      # Genuine 24-bit 96kHz → PASS
│   └── input192.flac     # 16-bit in 24-bit container → FAIL
│
├── CleanTranscoded/      # Honest bit-depth reductions
│   ├── input96_16bit.flac   # Genuinely 16-bit → PASS
│   └── input192_16bit.flac  # Genuinely 16-bit → PASS
│
├── Resample96/           # Sample rate changes from 96kHz
│   ├── input96_44.flac   # 96→44.1kHz (downsample) → PASS
│   ├── input96_48.flac   # 96→48kHz (downsample) → PASS
│   ├── input96_88.flac   # 96→88.2kHz (downsample) → PASS
│   ├── input96_176.flac  # 96→176.4kHz (upsample) → FAIL
│   └── input96_192.flac  # 96→192kHz (upsample) → FAIL
│
├── Upscale16/            # 16-bit padded to 24-bit
│   ├── output96_16bit.flac   # Fake 24-bit → FAIL
│   └── output192_16bit.flac  # Fake 24-bit → FAIL
│
├── Upscaled/             # Lossy codec transcodes
│   ├── input96_mp3.flac  # From MP3 → FAIL
│   ├── input96_m4a.flac  # From AAC → FAIL
│   ├── input96_opus.flac # From Opus → FAIL
│   └── input96_ogg.flac  # From Vorbis → FAIL
│
└── MasterScript/         # Complex transcoding chains (regression only)
    ├── test96_*.flac     # From genuine 24-bit source
    └── test192_*.flac    # From 16-bit source (all fail bit depth)
```

### Test Philosophy

| Scenario | Expected | Reason |
|----------|----------|--------|
| Genuine high-res master | ✅ PASS | Real data at claimed specs |
| Honest 16-bit transcode | ✅ PASS | File honestly claims 16-bit |
| Downsampled (96→44kHz) | ✅ PASS | Data discarded, not faked |
| Upsampled (44→96kHz) | ❌ FAIL | Interpolated samples = fake |
| 16→24 bit padding | ❌ FAIL | Zero-padded LSBs = fake |
| MP3→FLAC transcode | ❌ FAIL | Lossy artifacts present |

### Detection Categories

| Defect Type | Description |
|-------------|-------------|
| `BitDepthMismatch` | Claims higher bit depth than actual data |
| `Upsampled` | Sample rate increased via interpolation |
| `Mp3Transcode` | MP3 codec artifacts detected |
| `AacTranscode` | AAC codec artifacts detected |
| `OpusTranscode` | Opus codec artifacts detected |
| `OggVorbisTranscode` | Vorbis codec artifacts detected |

---

## Development

### Running Tests Locally

```bash
# Build
cargo build --release

# Run qualification tests (requires TestFiles/)
cargo test --test qualification_test -- --nocapture

# Run regression tests (requires full TestFiles/)
cargo test --test regression_test -- --nocapture
```

### Test Files Setup

1. Download from MinIO:
   ```bash
   mc cp myminio/audiocheckr/CompactTestFiles.zip .
   unzip CompactTestFiles.zip
   mv CompactTestFiles TestFiles
   ```

2. Or use the full set for regression:
   ```bash
   mc cp myminio/audiocheckr/TestFiles.zip .
   unzip TestFiles.zip
   ```

### Creating Compact Test Files

```powershell
# On Windows, from directory with full TestFiles/
.\Create-CompactTestFiles.ps1
# Creates CompactTestFiles.zip (~1.4GB)
```

---

## Jenkins Build Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TEST_TYPE_OVERRIDE` | `AUTO` | Force `QUALIFICATION` or `REGRESSION` |
| `SKIP_SONARQUBE` | `false` | Skip code quality analysis |
| `CLEAN_WORKSPACE_BEFORE` | `false` | Delete workspace before build |

### Storage Management

The pipeline automatically cleans up after each build:
- Deletes test files (1.4GB - 8.5GB)
- Cleans Cargo build cache (~2GB)
- Keeps only the release binary
- Retains last 10 builds (configurable)

---

## Troubleshooting

### SonarQube Quality Gate Timeout

**Symptom**: "Quality Gate skipped: Timeout has been exceeded"

**Fix**: Configure webhook in SonarQube (see SonarQube Setup section above)

### Test Files Not Found

**Symptom**: "TestFiles directory not found"

**Fix**: 
1. Check MinIO connectivity
2. Verify files exist: `mc ls myminio/audiocheckr/`
3. Upload test files if missing

### Tests Fail but Should Pass (or vice versa)

**Current Status**: Detector is in development. Expected failures:
- Upsampling detection (96→176/192kHz)
- Some codec detection at 96kHz

Track progress in the v0.2 branch.

### Disk Space Issues

**Fix**:
1. Reduce `numToKeepStr` in Jenkinsfile
2. Manually clean: `rm -rf /var/lib/jenkins/workspace/audiocheckr-ci`
3. Clean Cargo cache: `rm -rf ~/.cargo/registry/cache`

---

## License

See [LICENSE](LICENSE) file.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run qualification tests locally
4. Submit a pull request

---

## Current Test Results

| Test Category | Status | Notes |
|---------------|--------|-------|
| CleanOrigin | ✅ Working | |
| CleanTranscoded | ✅ Working | |
| Downsampling | ✅ Working | |
| Upsampling | ⚠️ In Progress | v0.2 |
| Bit Depth | ✅ Working | |
| MP3 Detection | ✅ Working | |
| AAC Detection | ⚠️ Partial | 192kHz works, 96kHz in progress |
| Opus Detection | ⚠️ Partial | 192kHz works, 96kHz in progress |
| Vorbis Detection | ⚠️ Partial | 192kHz works, 96kHz in progress |
