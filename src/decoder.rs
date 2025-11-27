// src/decoder.rs
//
// Audio decoding module with enhanced metadata extraction and validation.
// Uses Symphonia for format-agnostic decoding.

use anyhow::{Context, Result, bail};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use std::fs::File;
use std::path::Path;

/// Container for decoded audio data and metadata
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Interleaved samples normalized to [-1.0, 1.0]
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: usize,
    /// Bit depth as reported by the file (may not reflect actual precision)
    pub claimed_bit_depth: u32,
    /// Whether bit depth was inferred vs read from metadata
    pub bit_depth_inferred: bool,
    /// Duration in seconds
    pub duration_secs: f64,
    /// Original codec name
    pub codec_name: String,
    /// Container format
    pub format_name: String,
}

/// Decode audio file to floating-point samples
pub fn decode_audio(path: &Path) -> Result<AudioData> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open file: {}", path.display()))?;
    
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension() {
        hint.with_extension(ext.to_str().unwrap_or(""));
    }

    let meta_opts = MetadataOptions::default();
    let fmt_opts = FormatOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .context("Failed to probe file format - may be corrupted or unsupported")?;

    let format_name = format!("{:?}", probed.format.metadata());
    let mut format = probed.format;
    
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .context("No supported audio track found in file")?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate
        .context("File does not specify sample rate")?;
    
    let channels = track.codec_params.channels
        .map(|c| c.count())
        .unwrap_or(2);
    
    if channels == 0 {
        bail!("File reports 0 audio channels");
    }

    // Extract bit depth with inference tracking
    let (claimed_bit_depth, bit_depth_inferred) = 
        if let Some(bps) = track.codec_params.bits_per_sample {
            (bps, false)
        } else if let Some(bps) = track.codec_params.bits_per_coded_sample {
            (bps, true)
        } else {
            // Last resort: infer from codec
            let inferred = infer_bit_depth_from_codec(&track.codec_params);
            (inferred, true)
        };

    let codec_name = format!("{:?}", track.codec_params.codec);

    let dec_opts = DecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .context("Failed to create decoder - codec may not be supported")?;

    let mut sample_buf = None;
    let mut all_samples = Vec::new();
    
    // Pre-allocate based on duration hint if available
    if let Some(n_frames) = track.codec_params.n_frames {
        let estimated_samples = n_frames as usize * channels;
        all_samples.reserve(estimated_samples);
    }

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(ref e)) 
                if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(symphonia::core::errors::Error::ResetRequired) => {
                // Handle seek errors gracefully
                decoder.reset();
                continue;
            }
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                if sample_buf.is_none() {
                    let spec = *decoded.spec();
                    let duration = decoded.capacity() as u64;
                    sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                }

                if let Some(buf) = &mut sample_buf {
                    buf.copy_interleaved_ref(decoded);
                    all_samples.extend_from_slice(buf.samples());
                }
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => {
                // Skip corrupted frames
                continue;
            }
            Err(_) => continue,
        }
    }

    if all_samples.is_empty() {
        bail!("No audio samples could be decoded from file");
    }

    let duration_secs = all_samples.len() as f64 / (sample_rate as f64 * channels as f64);

    Ok(AudioData {
        samples: all_samples,
        sample_rate,
        channels,
        claimed_bit_depth,
        bit_depth_inferred,
        duration_secs,
        codec_name,
        format_name,
    })
}

/// Infer bit depth from codec type
fn infer_bit_depth_from_codec(params: &symphonia::core::codecs::CodecParameters) -> u32 {
    use symphonia::core::codecs::CodecType;
    
    match params.codec {
        // Lossless codecs - typically 16 or 24 bit
        codec if codec == CodecType::FLAC => 16,  // Conservative default
        codec if codec == CodecType::ALAC => 16,
        codec if codec == CodecType::WAVPACK => 16,
        
        // PCM variants
        codec if codec == CodecType::PCM_S16LE => 16,
        codec if codec == CodecType::PCM_S16BE => 16,
        codec if codec == CodecType::PCM_S24LE => 24,
        codec if codec == CodecType::PCM_S24BE => 24,
        codec if codec == CodecType::PCM_S32LE => 32,
        codec if codec == CodecType::PCM_S32BE => 32,
        codec if codec == CodecType::PCM_F32LE => 32,
        codec if codec == CodecType::PCM_F32BE => 32,
        
        // Lossy codecs - output as floating point, but source was limited
        codec if codec == CodecType::MP3 => 16,
        codec if codec == CodecType::AAC => 16,
        codec if codec == CodecType::VORBIS => 16,
        codec if codec == CodecType::OPUS => 16,
        
        // Default
        _ => 16,
    }
}

/// Extract mono channel from interleaved audio
pub fn extract_mono(audio: &AudioData) -> Vec<f32> {
    if audio.channels == 1 {
        audio.samples.clone()
    } else {
        audio.samples
            .chunks(audio.channels)
            .map(|chunk| {
                // Average all channels
                chunk.iter().sum::<f32>() / chunk.len() as f32
            })
            .collect()
    }
}

/// Extract left and right channels from stereo audio
pub fn extract_stereo(audio: &AudioData) -> Option<(Vec<f32>, Vec<f32>)> {
    if audio.channels < 2 {
        return None;
    }
    
    let left: Vec<f32> = audio.samples
        .chunks(audio.channels)
        .map(|chunk| chunk[0])
        .collect();
    
    let right: Vec<f32> = audio.samples
        .chunks(audio.channels)
        .map(|chunk| chunk[1])
        .collect();
    
    Some((left, right))
}

/// Compute Mid-Side representation from stereo
pub fn compute_mid_side(audio: &AudioData) -> Option<(Vec<f32>, Vec<f32>)> {
    let (left, right) = extract_stereo(audio)?;
    
    let mid: Vec<f32> = left.iter()
        .zip(&right)
        .map(|(l, r)| (l + r) * 0.5)
        .collect();
    
    let side: Vec<f32> = left.iter()
        .zip(&right)
        .map(|(l, r)| (l - r) * 0.5)
        .collect();
    
    Some((mid, side))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_mono() {
        let audio = AudioData {
            samples: vec![0.5, 0.3, 0.7, 0.1, -0.2, 0.4],
            sample_rate: 44100,
            channels: 2,
            claimed_bit_depth: 16,
            bit_depth_inferred: false,
            duration_secs: 0.0,
            codec_name: "test".to_string(),
            format_name: "test".to_string(),
        };
        
        let mono = extract_mono(&audio);
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.4).abs() < 0.001); // (0.5 + 0.3) / 2
    }

    #[test]
    fn test_mid_side() {
        let audio = AudioData {
            samples: vec![1.0, 0.0, 0.0, 1.0],  // L=1,R=0 then L=0,R=1
            sample_rate: 44100,
            channels: 2,
            claimed_bit_depth: 16,
            bit_depth_inferred: false,
            duration_secs: 0.0,
            codec_name: "test".to_string(),
            format_name: "test".to_string(),
        };
        
        let (mid, side) = compute_mid_side(&audio).unwrap();
        assert_eq!(mid.len(), 2);
        assert!((mid[0] - 0.5).abs() < 0.001);
        assert!((side[0] - 0.5).abs() < 0.001);
    }
}
