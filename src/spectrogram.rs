// src/spectrogram.rs
//
// Spectrogram generation with multiple visualization options.
// Supports mel scale, linear scale, and enhanced visual features.

use anyhow::Result;
use image::{ImageBuffer, Rgb, RgbImage};
use imageproc::drawing::draw_text_mut;
use rusttype::{Font, Scale};
use std::path::Path;
use crate::decoder::AudioData;
use crate::dsp::{FftProcessor, WindowType, pre_emphasis};

/// Spectrogram configuration
pub struct SpectrogramConfig {
    /// FFT window size
    pub window_size: usize,
    /// Hop size (overlap = window_size - hop_size)
    pub hop_size: usize,
    /// Window function type
    pub window_type: WindowType,
    /// Apply pre-emphasis filter
    pub pre_emphasis: bool,
    /// Pre-emphasis coefficient
    pub pre_emphasis_coeff: f32,
    /// Number of mel bands (for mel spectrogram)
    pub num_mel_bands: usize,
    /// Minimum dB for display range
    pub min_db: f32,
    /// Maximum dB for display range
    pub max_db: f32,
    /// Colormap to use
    pub colormap: Colormap,
    /// Image width in pixels (0 = auto based on duration)
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Margin sizes
    pub margin_left: u32,
    pub margin_right: u32,
    pub margin_top: u32,
    pub margin_bottom: u32,
}

impl Default for SpectrogramConfig {
    fn default() -> Self {
        SpectrogramConfig {
            window_size: 2048,
            hop_size: 512,
            window_type: WindowType::Hann,
            pre_emphasis: true,
            pre_emphasis_coeff: 0.97,
            num_mel_bands: 256,
            min_db: -80.0,
            max_db: 0.0,
            colormap: Colormap::Viridis,
            width: 0,
            height: 600,
            margin_left: 100,
            margin_right: 120,
            margin_top: 80,
            margin_bottom: 80,
        }
    }
}

/// Colormap options
#[derive(Debug, Clone, Copy)]
pub enum Colormap {
    Viridis,
    Magma,
    Inferno,
    Plasma,
    Grayscale,
}

/// Generate spectrogram image with default settings
pub fn generate_spectrogram_image(
    audio: &AudioData,
    output_path: &Path,
    use_linear_scale: bool,
    full_length: bool,
) -> Result<()> {
    let config = SpectrogramConfig::default();
    
    if use_linear_scale {
        generate_linear_spectrogram(audio, output_path, &config, full_length)
    } else {
        generate_mel_spectrogram(audio, output_path, &config, full_length)
    }
}

/// Generate mel-scale spectrogram
pub fn generate_mel_spectrogram(
    audio: &AudioData,
    output_path: &Path,
    config: &SpectrogramConfig,
    full_length: bool,
) -> Result<()> {
    let mono = crate::decoder::extract_mono(audio);
    
    // Apply pre-emphasis if enabled
    let processed = if config.pre_emphasis {
        pre_emphasis(&mono, config.pre_emphasis_coeff)
    } else {
        mono
    };
    
    // Limit duration
    let max_duration_secs = if full_length { f32::MAX } else { 15.0 };
    let max_samples = (audio.sample_rate as f32 * max_duration_secs) as usize;
    let samples = if processed.len() > max_samples {
        &processed[..max_samples]
    } else {
        &processed
    };
    
    // Compute spectrogram
    let spectrogram = compute_spectrogram(samples, config);
    
    // Convert to mel scale
    let mel_filters = create_mel_filterbank(
        config.num_mel_bands,
        config.window_size / 2,
        audio.sample_rate,
        20.0,
        audio.sample_rate as f32 / 2.0,
    );
    
    let mel_spectrogram = apply_mel_filterbank_to_spectrogram(&spectrogram, &mel_filters);
    
    // Convert to dB
    let db_spectrogram = to_decibels(&mel_spectrogram, config.min_db, config.max_db);
    
    // Render image
    let duration = samples.len() as f32 / audio.sample_rate as f32;
    render_spectrogram_image(
        &db_spectrogram,
        output_path,
        config,
        audio.sample_rate,
        duration,
        true,  // is_mel
    )
}

/// Generate linear-scale spectrogram
pub fn generate_linear_spectrogram(
    audio: &AudioData,
    output_path: &Path,
    config: &SpectrogramConfig,
    full_length: bool,
) -> Result<()> {
    let mono = crate::decoder::extract_mono(audio);
    
    let processed = if config.pre_emphasis {
        pre_emphasis(&mono, config.pre_emphasis_coeff)
    } else {
        mono
    };
    
    let max_duration_secs = if full_length { f32::MAX } else { 15.0 };
    let max_samples = (audio.sample_rate as f32 * max_duration_secs) as usize;
    let samples = if processed.len() > max_samples {
        &processed[..max_samples]
    } else {
        &processed
    };
    
    let spectrogram = compute_spectrogram(samples, config);
    let db_spectrogram = to_decibels(&spectrogram, config.min_db, config.max_db);
    
    let duration = samples.len() as f32 / audio.sample_rate as f32;
    render_spectrogram_image(
        &db_spectrogram,
        output_path,
        config,
        audio.sample_rate,
        duration,
        false,  // not mel
    )
}

/// Compute raw spectrogram (magnitude in each bin)
fn compute_spectrogram(samples: &[f32], config: &SpectrogramConfig) -> Vec<Vec<f32>> {
    let mut fft = FftProcessor::new(config.window_size, config.window_type);
    let num_bins = config.window_size / 2;
    let num_frames = samples.len().saturating_sub(config.window_size) / config.hop_size;
    
    if num_frames == 0 {
        return vec![vec![0.0; 1]; num_bins];
    }
    
    let mut spectrogram = vec![vec![0.0f32; num_frames]; num_bins];
    
    for frame_idx in 0..num_frames {
        let start = frame_idx * config.hop_size;
        if start + config.window_size > samples.len() {
            break;
        }
        
        let frame = &samples[start..start + config.window_size];
        let magnitudes = fft.magnitude_spectrum(frame);
        
        for (bin, &mag) in magnitudes.iter().enumerate() {
            if bin < num_bins {
                spectrogram[bin][frame_idx] = mag;
            }
        }
    }
    
    spectrogram
}

/// Create mel filterbank
fn create_mel_filterbank(
    num_mel_bins: usize,
    num_fft_bins: usize,
    sample_rate: u32,
    min_freq: f32,
    max_freq: f32,
) -> Vec<Vec<f32>> {
    let min_mel = hz_to_mel(min_freq);
    let max_mel = hz_to_mel(max_freq);
    
    let mel_points: Vec<f32> = (0..=num_mel_bins + 1)
        .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (num_mel_bins + 1) as f32)
        .collect();
    
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<f32> = hz_points.iter()
        .map(|&f| num_fft_bins as f32 * f / (sample_rate as f32 / 2.0))
        .collect();
    
    let mut filters = vec![vec![0.0; num_fft_bins]; num_mel_bins];
    
    for i in 0..num_mel_bins {
        let left = bin_points[i];
        let center = bin_points[i + 1];
        let right = bin_points[i + 2];
        
        for j in 0..num_fft_bins {
            let j_f = j as f32;
            if j_f >= left && j_f <= center && center > left {
                filters[i][j] = (j_f - left) / (center - left);
            } else if j_f > center && j_f <= right && right > center {
                filters[i][j] = (right - j_f) / (right - center);
            }
        }
    }
    
    filters
}

/// Apply mel filterbank to spectrogram
fn apply_mel_filterbank_to_spectrogram(
    spectrogram: &[Vec<f32>],
    filters: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let num_mel_bins = filters.len();
    let num_frames = spectrogram.get(0).map(|f| f.len()).unwrap_or(0);
    
    let mut mel_spec = vec![vec![0.0f32; num_frames]; num_mel_bins];
    
    for (mel_idx, filter) in filters.iter().enumerate() {
        for frame_idx in 0..num_frames {
            let mut sum = 0.0f32;
            for (bin_idx, &weight) in filter.iter().enumerate() {
                if bin_idx < spectrogram.len() && frame_idx < spectrogram[bin_idx].len() {
                    // Use power (magnitude squared) for mel
                    sum += spectrogram[bin_idx][frame_idx].powi(2) * weight;
                }
            }
            mel_spec[mel_idx][frame_idx] = sum.sqrt();  // Back to magnitude
        }
    }
    
    mel_spec
}

/// Convert to decibels with normalization
fn to_decibels(spectrogram: &[Vec<f32>], min_db: f32, max_db: f32) -> Vec<Vec<f32>> {
    // Find max value
    let max_val = spectrogram.iter()
        .flat_map(|row| row.iter())
        .cloned()
        .fold(0.0f32, f32::max)
        .max(1e-10);
    
    spectrogram.iter()
        .map(|row| {
            row.iter()
                .map(|&val| {
                    let db = if val > 0.0 {
                        20.0 * (val / max_val).log10()
                    } else {
                        min_db
                    };
                    ((db - min_db) / (max_db - min_db)).clamp(0.0, 1.0)
                })
                .collect()
        })
        .collect()
}

/// Render spectrogram to image file
fn render_spectrogram_image(
    spectrogram: &[Vec<f32>],
    output_path: &Path,
    config: &SpectrogramConfig,
    sample_rate: u32,
    duration: f32,
    is_mel: bool,
) -> Result<()> {
    let num_bins = spectrogram.len();
    let num_frames = spectrogram.get(0).map(|f| f.len()).unwrap_or(1);
    
    // Calculate image dimensions
    let spec_width = if config.width > 0 {
        config.width
    } else {
        (num_frames * 2).max(1200) as u32
    };
    let spec_height = config.height;
    
    let img_width = config.margin_left + spec_width + config.margin_right;
    let img_height = config.margin_top + spec_height + config.margin_bottom;
    
    let mut img: RgbImage = ImageBuffer::from_pixel(img_width, img_height, Rgb([255u8, 255u8, 255u8]));
    
    // Draw spectrogram with bilinear interpolation
    for x in 0..spec_width {
        for y in 0..spec_height {
            let frame_f = (x as f32 / spec_width as f32) * num_frames as f32;
            let bin_f = (1.0 - y as f32 / spec_height as f32) * num_bins as f32;
            
            let frame_idx = (frame_f.floor() as usize).min(num_frames.saturating_sub(1));
            let bin_idx = (bin_f.floor() as usize).min(num_bins.saturating_sub(1));
            
            // Bilinear interpolation
            let value = if frame_idx < num_frames.saturating_sub(1) && bin_idx < num_bins.saturating_sub(1) {
                let fx = frame_f - frame_idx as f32;
                let fy = bin_f - bin_idx as f32;
                
                let v00 = spectrogram[bin_idx][frame_idx];
                let v10 = spectrogram[bin_idx][frame_idx + 1];
                let v01 = spectrogram[bin_idx + 1][frame_idx];
                let v11 = spectrogram[bin_idx + 1][frame_idx + 1];
                
                let v0 = v00 * (1.0 - fx) + v10 * fx;
                let v1 = v01 * (1.0 - fx) + v11 * fx;
                v0 * (1.0 - fy) + v1 * fy
            } else {
                spectrogram.get(bin_idx)
                    .and_then(|row| row.get(frame_idx))
                    .copied()
                    .unwrap_or(0.0)
            };
            
            let color = value_to_color(value, config.colormap);
            img.put_pixel(
                config.margin_left + x,
                config.margin_top + y,
                Rgb([color.0, color.1, color.2]),
            );
        }
    }
    
    // Load font
    let font_data = include_bytes!("../fonts/DejaVuSans.ttf");
    let font = Font::try_from_bytes(font_data as &[u8])
        .ok_or_else(|| anyhow::anyhow!("Failed to load font"))?;
    
    // Draw title
    let title = if is_mel { "Mel Spectrogram" } else { "Linear Spectrogram" };
    let title_scale = Scale::uniform(28.0);
    draw_text_mut(
        &mut img,
        Rgb([0, 0, 0]),
        (config.margin_left + spec_width / 2 - 80) as i32,
        25,
        title_scale,
        &font,
        title,
    );
    
    // Draw axes
    draw_axes(&mut img, config.margin_left, config.margin_top, spec_width, spec_height);
    
    // Draw time labels
    draw_time_labels(&mut img, config, spec_width, spec_height, duration, &font);
    
    // Draw frequency labels
    if is_mel {
        draw_mel_frequency_labels(&mut img, config, spec_height, sample_rate, &font);
    } else {
        draw_linear_frequency_labels(&mut img, config, spec_height, sample_rate, &font);
    }
    
    // Draw colorbar
    draw_colorbar(&mut img, config, spec_height, &font);
    
    img.save(output_path)?;
    Ok(())
}

/// Convert normalized value to RGB color
fn value_to_color(value: f32, colormap: Colormap) -> (u8, u8, u8) {
    let v = value.clamp(0.0, 1.0);
    
    match colormap {
        Colormap::Viridis => viridis_color(v),
        Colormap::Magma => magma_color(v),
        Colormap::Inferno => inferno_color(v),
        Colormap::Plasma => plasma_color(v),
        Colormap::Grayscale => {
            let g = (v * 255.0) as u8;
            (g, g, g)
        }
    }
}

/// Viridis colormap
fn viridis_color(v: f32) -> (u8, u8, u8) {
    if v < 0.25 {
        let t = v / 0.25;
        let r = (68.0 + t * 12.0) as u8;
        let g = (1.0 + t * 38.0) as u8;
        let b = (84.0 + t * 56.0) as u8;
        (r, g, b)
    } else if v < 0.5 {
        let t = (v - 0.25) / 0.25;
        let r = (80.0 - t * 38.0) as u8;
        let g = (39.0 + t * 107.0) as u8;
        let b = (140.0 - t * 8.0) as u8;
        (r, g, b)
    } else if v < 0.75 {
        let t = (v - 0.5) / 0.25;
        let r = (42.0 + t * 81.0) as u8;
        let g = (146.0 + t * 52.0) as u8;
        let b = (132.0 - t * 69.0) as u8;
        (r, g, b)
    } else {
        let t = (v - 0.75) / 0.25;
        let r = (123.0 + t * 130.0) as u8;
        let g = (198.0 + t * 27.0) as u8;
        let b = (63.0 - t * 18.0) as u8;
        (r, g, b)
    }
}

/// Magma colormap
fn magma_color(v: f32) -> (u8, u8, u8) {
    if v < 0.25 {
        let t = v / 0.25;
        ((t * 50.0) as u8, (t * 15.0) as u8, (20.0 + t * 50.0) as u8)
    } else if v < 0.5 {
        let t = (v - 0.25) / 0.25;
        ((50.0 + t * 130.0) as u8, (15.0 + t * 25.0) as u8, (70.0 + t * 30.0) as u8)
    } else if v < 0.75 {
        let t = (v - 0.5) / 0.25;
        ((180.0 + t * 50.0) as u8, (40.0 + t * 80.0) as u8, (100.0 - t * 20.0) as u8)
    } else {
        let t = (v - 0.75) / 0.25;
        ((230.0 + t * 25.0) as u8, (120.0 + t * 135.0) as u8, (80.0 + t * 100.0) as u8)
    }
}

/// Inferno colormap
fn inferno_color(v: f32) -> (u8, u8, u8) {
    if v < 0.25 {
        let t = v / 0.25;
        ((t * 40.0) as u8, (t * 10.0) as u8, (20.0 + t * 60.0) as u8)
    } else if v < 0.5 {
        let t = (v - 0.25) / 0.25;
        ((40.0 + t * 120.0) as u8, (10.0 + t * 20.0) as u8, (80.0 - t * 20.0) as u8)
    } else if v < 0.75 {
        let t = (v - 0.5) / 0.25;
        ((160.0 + t * 70.0) as u8, (30.0 + t * 100.0) as u8, (60.0 - t * 40.0) as u8)
    } else {
        let t = (v - 0.75) / 0.25;
        ((230.0 + t * 22.0) as u8, (130.0 + t * 115.0) as u8, (20.0 + t * 80.0) as u8)
    }
}

/// Plasma colormap
fn plasma_color(v: f32) -> (u8, u8, u8) {
    if v < 0.25 {
        let t = v / 0.25;
        ((15.0 + t * 80.0) as u8, (5.0 + t * 5.0) as u8, (105.0 + t * 50.0) as u8)
    } else if v < 0.5 {
        let t = (v - 0.25) / 0.25;
        ((95.0 + t * 90.0) as u8, (10.0 + t * 20.0) as u8, (155.0 - t * 30.0) as u8)
    } else if v < 0.75 {
        let t = (v - 0.5) / 0.25;
        ((185.0 + t * 50.0) as u8, (30.0 + t * 80.0) as u8, (125.0 - t * 50.0) as u8)
    } else {
        let t = (v - 0.75) / 0.25;
        ((235.0 + t * 15.0) as u8, (110.0 + t * 110.0) as u8, (75.0 - t * 65.0) as u8)
    }
}

/// Draw axes
fn draw_axes(img: &mut RgbImage, x: u32, y: u32, width: u32, height: u32) {
    let black = Rgb([0u8, 0u8, 0u8]);
    
    for thickness in 0..2 {
        for dy in 0..=height {
            if x + thickness < img.width() {
                img.put_pixel(x + thickness, y + dy, black);
            }
        }
        for dx in 0..=width {
            if y + height + thickness < img.height() {
                img.put_pixel(x + dx, y + height + thickness, black);
            }
        }
    }
}

/// Draw time labels
fn draw_time_labels(
    img: &mut RgbImage,
    config: &SpectrogramConfig,
    spec_width: u32,
    spec_height: u32,
    duration: f32,
    font: &Font,
) {
    let num_labels = 6;
    let scale = Scale::uniform(16.0);
    
    for i in 0..=num_labels {
        let x = config.margin_left + (i as f32 / num_labels as f32 * spec_width as f32) as u32;
        let time = i as f32 / num_labels as f32 * duration;
        let label = format!("{:.1}", time);
        
        draw_text_mut(
            img,
            Rgb([0, 0, 0]),
            (x as i32).saturating_sub(15),
            (config.margin_top + spec_height + 15) as i32,
            scale,
            font,
            &label,
        );
        
        // Tick mark
        for dy in 0..8 {
            if config.margin_top + spec_height + dy < img.height() {
                img.put_pixel(x, config.margin_top + spec_height + dy, Rgb([0, 0, 0]));
            }
        }
    }
    
    // Axis label
    let label_scale = Scale::uniform(20.0);
    draw_text_mut(
        img,
        Rgb([0, 0, 0]),
        (config.margin_left + spec_width / 2 - 25) as i32,
        (config.margin_top + spec_height + 50) as i32,
        label_scale,
        font,
        "Time (s)",
    );
}

/// Draw mel frequency labels
fn draw_mel_frequency_labels(
    img: &mut RgbImage,
    config: &SpectrogramConfig,
    spec_height: u32,
    sample_rate: u32,
    font: &Font,
) {
    let scale = Scale::uniform(15.0);
    let freq_markers = [100, 250, 500, 1000, 2000, 4000, 8000, 16000];
    let max_freq = sample_rate as f32 / 2.0;
    let max_mel = hz_to_mel(max_freq);
    
    for &freq in &freq_markers {
        if freq as f32 > max_freq {
            continue;
        }
        
        let mel = hz_to_mel(freq as f32);
        let y = config.margin_top + spec_height - (mel / max_mel * spec_height as f32) as u32;
        
        if y >= config.margin_top && y < config.margin_top + spec_height {
            let label = if freq >= 1000 {
                format!("{}k", freq / 1000)
            } else {
                format!("{}", freq)
            };
            
            draw_text_mut(
                img,
                Rgb([0, 0, 0]),
                15,
                (y as i32).saturating_sub(8),
                scale,
                font,
                &label,
            );
            
            // Tick mark
            for dx in 0..8 {
                if config.margin_left > dx {
                    img.put_pixel(config.margin_left - dx, y, Rgb([0, 0, 0]));
                }
            }
        }
    }
    
    // Axis label
    let label_scale = Scale::uniform(18.0);
    draw_text_mut(
        img,
        Rgb([0, 0, 0]),
        10,
        (config.margin_top + spec_height / 2 - 40) as i32,
        label_scale,
        font,
        "Hz",
    );
}

/// Draw linear frequency labels
fn draw_linear_frequency_labels(
    img: &mut RgbImage,
    config: &SpectrogramConfig,
    spec_height: u32,
    sample_rate: u32,
    font: &Font,
) {
    let scale = Scale::uniform(15.0);
    let nyquist = sample_rate / 2;
    let freq_markers: Vec<u32> = (0..=nyquist).step_by(2000).collect();
    
    for &freq in &freq_markers {
        let y = config.margin_top + spec_height 
            - (freq as f32 / nyquist as f32 * spec_height as f32) as u32;
        
        if y >= config.margin_top && y < config.margin_top + spec_height {
            let label = if freq >= 1000 {
                format!("{}k", freq / 1000)
            } else {
                format!("{}", freq)
            };
            
            draw_text_mut(
                img,
                Rgb([0, 0, 0]),
                15,
                (y as i32).saturating_sub(8),
                scale,
                font,
                &label,
            );
            
            for dx in 0..8 {
                if config.margin_left > dx {
                    img.put_pixel(config.margin_left - dx, y, Rgb([0, 0, 0]));
                }
            }
        }
    }
    
    let label_scale = Scale::uniform(18.0);
    draw_text_mut(
        img,
        Rgb([0, 0, 0]),
        10,
        (config.margin_top + spec_height / 2 - 40) as i32,
        label_scale,
        font,
        "Hz",
    );
}

/// Draw colorbar
fn draw_colorbar(
    img: &mut RgbImage,
    config: &SpectrogramConfig,
    spec_height: u32,
    font: &Font,
) {
    let bar_x = config.margin_left + (img.width() - config.margin_left - config.margin_right) + 25;
    let bar_width = 40;
    let bar_height = spec_height;
    
    // Draw gradient
    for i in 0..bar_height {
        let value = i as f32 / bar_height as f32;
        let color = value_to_color(value, config.colormap);
        for j in 0..bar_width {
            if bar_x + j < img.width() {
                img.put_pixel(bar_x + j, config.margin_top + bar_height - 1 - i, 
                    Rgb([color.0, color.1, color.2]));
            }
        }
    }
    
    // Border
    let black = Rgb([0u8, 0u8, 0u8]);
    for i in 0..bar_height {
        if bar_x < img.width() {
            img.put_pixel(bar_x, config.margin_top + i, black);
        }
        if bar_x + bar_width - 1 < img.width() {
            img.put_pixel(bar_x + bar_width - 1, config.margin_top + i, black);
        }
    }
    for j in 0..bar_width {
        if bar_x + j < img.width() {
            img.put_pixel(bar_x + j, config.margin_top, black);
            img.put_pixel(bar_x + j, config.margin_top + bar_height - 1, black);
        }
    }
    
    // Labels
    let scale = Scale::uniform(16.0);
    let db_labels = [0, -20, -40, -60, -80];
    
    for &db in &db_labels {
        let normalized = (db as f32 - config.min_db) / (config.max_db - config.min_db);
        let y = config.margin_top + bar_height - (normalized * bar_height as f32) as u32;
        
        draw_text_mut(
            img,
            Rgb([0, 0, 0]),
            (bar_x + bar_width + 8) as i32,
            (y as i32).saturating_sub(8),
            scale,
            font,
            &format!("{}", db),
        );
        
        // Tick
        for dx in 0..6 {
            if bar_x + bar_width + dx < img.width() && y < img.height() {
                img.put_pixel(bar_x + bar_width + dx, y, black);
            }
        }
    }
    
    // dB label
    let label_scale = Scale::uniform(18.0);
    draw_text_mut(
        img,
        Rgb([0, 0, 0]),
        (bar_x + bar_width + 8) as i32,
        config.margin_top.saturating_sub(25) as i32,
        label_scale,
        font,
        "dB",
    );
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}
