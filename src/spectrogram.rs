// src/spectrogram.rs

use anyhow::Result;
use image::{ImageBuffer, Rgb, RgbImage};
use imageproc::drawing::{draw_text_mut};
use rustfft::{FftPlanner, num_complex::Complex};
use rusttype::{Font, Scale};
use std::path::Path;
use crate::decoder::AudioData;

pub fn generate_spectrogram_image(
    audio: &AudioData,
    output_path: &Path,
    use_linear_scale: bool,
    full_length: bool
) -> Result<()> {
    let window_size = 2048;
    let hop_size = window_size / 4;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(window_size);

    // Extract mono channel
    let mono: Vec<f32> = audio.samples
        .chunks(audio.channels)
        .map(|chunk| chunk[0])
        .collect();

    // Limit to first 15 seconds if not full_length
    let max_duration_secs = if full_length { f32::MAX } else { 15.0 };
    let max_samples = (audio.sample_rate as f32 * max_duration_secs) as usize;
    let samples_to_process = mono.len().min(max_samples);
    let mono_trimmed = &mono[..samples_to_process];

    let num_frames = (mono_trimmed.len().saturating_sub(window_size)) / hop_size;
    let num_bins = window_size / 2;
    let mut spectrogram = vec![vec![0.0f32; num_frames]; num_bins];

    // Compute FFT
    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        let end = start + window_size;
        if end > mono_trimmed.len() {
            break;
        }

        let mut buffer: Vec<Complex<f32>> = mono_trimmed[start..end]
            .iter()
            .enumerate()
            .map(|(i, &sample)| {
                let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / window_size as f32).cos());
                Complex::new(sample * window, 0.0)
            })
            .collect();

        fft.process(&mut buffer);

        for (bin_idx, complex) in buffer[..num_bins].iter().enumerate() {
            let magnitude = (complex.re * complex.re + complex.im * complex.im).sqrt();
            spectrogram[bin_idx][frame_idx] = magnitude;
        }
    }

    let duration = samples_to_process as f32 / audio.sample_rate as f32;

    if use_linear_scale {
        render_linear_spectrogram(&spectrogram, output_path, audio.sample_rate, duration)
    } else {
        render_mel_spectrogram(&spectrogram, output_path, audio.sample_rate, num_bins, duration)
    }
}

fn render_mel_spectrogram(
    spectrogram: &[Vec<f32>],
    output_path: &Path,
    sample_rate: u32,
    num_bins: usize,
    duration: f32
) -> Result<()> {
    let num_mel_bins = 300;
    let num_frames = spectrogram[0].len();
    
    let mel_filters = create_mel_filterbank(num_mel_bins, num_bins, sample_rate);
    
    let mut mel_spec = vec![vec![0.0f32; num_frames]; num_mel_bins];
    for (mel_idx, filter) in mel_filters.iter().enumerate() {
        for frame_idx in 0..num_frames {
            let mut sum = 0.0;
            for (bin_idx, &weight) in filter.iter().enumerate() {
                sum += spectrogram[bin_idx][frame_idx] * weight;
            }
            mel_spec[mel_idx][frame_idx] = sum;
        }
    }

    let db_spec = normalize_to_db(&mel_spec);

    // High quality image dimensions
    let margin_left = 100u32;
    let margin_right = 120u32;
    let margin_top = 80u32;
    let margin_bottom = 80u32;
    
    // Higher resolution for better quality
    let spec_width = (num_frames * 2).max(1200) as u32;
    let spec_height = 600u32;
    
    let img_width = margin_left + spec_width + margin_right;
    let img_height = margin_top + spec_height + margin_bottom;
    
    let mut img = ImageBuffer::from_pixel(img_width, img_height, Rgb([255u8, 255u8, 255u8]));
    
    // Draw spectrogram with interpolation for smoother appearance
    for x in 0..spec_width {
        for y in 0..spec_height {
            let frame_f = (x as f32 / spec_width as f32) * num_frames as f32;
            let mel_f = (1.0 - y as f32 / spec_height as f32) * num_mel_bins as f32;
            
            // Bilinear interpolation for smoother image
            let frame_idx = frame_f.floor() as usize;
            let mel_idx = mel_f.floor() as usize;
            
            if frame_idx < num_frames - 1 && mel_idx < num_mel_bins - 1 {
                let fx = frame_f - frame_idx as f32;
                let fy = mel_f - mel_idx as f32;
                
                let v00 = db_spec[mel_idx][frame_idx];
                let v10 = db_spec[mel_idx][frame_idx + 1];
                let v01 = db_spec[mel_idx + 1][frame_idx];
                let v11 = db_spec[mel_idx + 1][frame_idx + 1];
                
                let v0 = v00 * (1.0 - fx) + v10 * fx;
                let v1 = v01 * (1.0 - fx) + v11 * fx;
                let db_value = v0 * (1.0 - fy) + v1 * fy;
                
                let color = db_to_color(db_value);
                img.put_pixel(margin_left + x, margin_top + y, Rgb([color.0, color.1, color.2]));
            } else if frame_idx < num_frames && mel_idx < num_mel_bins {
                let db_value = db_spec[mel_idx][frame_idx];
                let color = db_to_color(db_value);
                img.put_pixel(margin_left + x, margin_top + y, Rgb([color.0, color.1, color.2]));
            }
        }
    }
    
    // Load font
    let font_data = include_bytes!("../fonts/DejaVuSans.ttf");
    let font = Font::try_from_bytes(font_data as &[u8]).unwrap();
    
    // Draw title
    let title_scale = Scale::uniform(28.0);
    draw_text_mut(&mut img, Rgb([0, 0, 0]), 
        (margin_left + spec_width / 2 - 100) as i32, 25, 
        title_scale, &font, "Mel Spectrogram");
    
    // Draw axes
    draw_axes(&mut img, margin_left, margin_top, spec_width, spec_height);
    
    // Draw time labels (X-axis)
    let num_time_labels = 6;
    let time_scale = Scale::uniform(16.0);
    for i in 0..=num_time_labels {
        let x = margin_left + (i as f32 / num_time_labels as f32 * spec_width as f32) as u32;
        let time = i as f32 / num_time_labels as f32 * duration;
        let label = format!("{:.1}", time);
        draw_text_mut(&mut img, Rgb([0, 0, 0]), 
            (x as i32).saturating_sub(15), (margin_top + spec_height + 15) as i32, 
            time_scale, &font, &label);
        
        // Draw tick
        for dy in 0..8 {
            img.put_pixel(x, margin_top + spec_height + dy, Rgb([0, 0, 0]));
        }
    }
    
    // Draw "Time" label
    let axis_scale = Scale::uniform(20.0);
    draw_text_mut(&mut img, Rgb([0, 0, 0]), 
        (margin_left + spec_width / 2 - 25) as i32, (margin_top + spec_height + 50) as i32, 
        axis_scale, &font, "Time (s)");
    
    // Draw frequency labels (Y-axis)
    let freq_scale = Scale::uniform(15.0);
    let freq_markers = [100, 250, 500, 1000, 2000, 4000, 8000, 16000];
    let max_freq = (sample_rate / 2) as f32;
    
    for &freq in &freq_markers {
        if freq as f32 > max_freq {
            continue;
        }
        
        let mel = hz_to_mel(freq as f32);
        let max_mel = hz_to_mel(max_freq);
        let y = margin_top + spec_height - (mel / max_mel * spec_height as f32) as u32;
        
        if y >= margin_top && y < margin_top + spec_height {
            let label = if freq >= 1000 {
                format!("{}k", freq / 1000)
            } else {
                format!("{}", freq)
            };
            draw_text_mut(&mut img, Rgb([0, 0, 0]), 
                15, (y as i32).saturating_sub(8), 
                freq_scale, &font, &label);
            
            // Draw tick
            for dx in 0..8 {
                if margin_left > dx {
                    img.put_pixel(margin_left - dx, y, Rgb([0, 0, 0]));
                }
            }
        }
    }
    
    // Draw "Frequency" label (rotated effect using vertical text)
    draw_text_mut(&mut img, Rgb([0, 0, 0]), 
        10, (margin_top + spec_height / 2 - 40) as i32, 
        axis_scale, &font, "Hz");
    
    // Draw dB color scale
    draw_db_colorscale(&mut img, margin_left + spec_width + 25, margin_top, 40, spec_height, &font);
    
    img.save(output_path)?;
    Ok(())
}

fn render_linear_spectrogram(
    spectrogram: &[Vec<f32>],
    output_path: &Path,
    sample_rate: u32,
    duration: f32
) -> Result<()> {
    let num_bins = spectrogram.len();
    let num_frames = spectrogram[0].len();
    
    let db_spec = normalize_to_db(spectrogram);
    
    let margin_left = 100u32;
    let margin_right = 120u32;
    let margin_top = 80u32;
    let margin_bottom = 80u32;
    
    let spec_width = (num_frames * 2).max(1200) as u32;
    let spec_height = 800u32;
    
    let img_width = margin_left + spec_width + margin_right;
    let img_height = margin_top + spec_height + margin_bottom;
    
    let mut img = ImageBuffer::from_pixel(img_width, img_height, Rgb([255u8, 255u8, 255u8]));
    
    // Draw with interpolation
    for x in 0..spec_width {
        for y in 0..spec_height {
            let frame_f = (x as f32 / spec_width as f32) * num_frames as f32;
            let bin_f = (1.0 - y as f32 / spec_height as f32) * num_bins as f32;
            
            let frame_idx = frame_f.floor() as usize;
            let bin_idx = bin_f.floor() as usize;
            
            if frame_idx < num_frames - 1 && bin_idx < num_bins - 1 {
                let fx = frame_f - frame_idx as f32;
                let fy = bin_f - bin_idx as f32;
                
                let v00 = db_spec[bin_idx][frame_idx];
                let v10 = db_spec[bin_idx][frame_idx + 1];
                let v01 = db_spec[bin_idx + 1][frame_idx];
                let v11 = db_spec[bin_idx + 1][frame_idx + 1];
                
                let v0 = v00 * (1.0 - fx) + v10 * fx;
                let v1 = v01 * (1.0 - fx) + v11 * fx;
                let db_value = v0 * (1.0 - fy) + v1 * fy;
                
                let color = db_to_color(db_value);
                img.put_pixel(margin_left + x, margin_top + y, Rgb([color.0, color.1, color.2]));
            } else if frame_idx < num_frames && bin_idx < num_bins {
                let db_value = db_spec[bin_idx][frame_idx];
                let color = db_to_color(db_value);
                img.put_pixel(margin_left + x, margin_top + y, Rgb([color.0, color.1, color.2]));
            }
        }
    }
    
    let font_data = include_bytes!("../fonts/DejaVuSans.ttf");
    let font = Font::try_from_bytes(font_data as &[u8]).unwrap();
    
    let title_scale = Scale::uniform(28.0);
    draw_text_mut(&mut img, Rgb([0, 0, 0]), 
        (margin_left + spec_width / 2 - 120) as i32, 25, 
        title_scale, &font, "Linear Spectrogram");
    
    draw_axes(&mut img, margin_left, margin_top, spec_width, spec_height);
    
    // Time labels
    let num_time_labels = 6;
    let time_scale = Scale::uniform(16.0);
    for i in 0..=num_time_labels {
        let x = margin_left + (i as f32 / num_time_labels as f32 * spec_width as f32) as u32;
        let time = i as f32 / num_time_labels as f32 * duration;
        draw_text_mut(&mut img, Rgb([0, 0, 0]), 
            (x as i32).saturating_sub(15), (margin_top + spec_height + 15) as i32, 
            time_scale, &font, &format!("{:.1}", time));
        
        for dy in 0..8 {
            img.put_pixel(x, margin_top + spec_height + dy, Rgb([0, 0, 0]));
        }
    }
    
    let axis_scale = Scale::uniform(20.0);
    draw_text_mut(&mut img, Rgb([0, 0, 0]), 
        (margin_left + spec_width / 2 - 25) as i32, (margin_top + spec_height + 50) as i32, 
        axis_scale, &font, "Time (s)");
    
    // Frequency labels
    let freq_scale = Scale::uniform(15.0);
    let nyquist = sample_rate / 2;
    let freq_markers = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000];
    
    for &freq in &freq_markers {
        if freq > nyquist {
            continue;
        }
        
        let y = margin_top + spec_height - (freq as f32 / nyquist as f32 * spec_height as f32) as u32;
        
        if y >= margin_top && y < margin_top + spec_height {
            let label = if freq >= 1000 {
                format!("{}k", freq / 1000)
            } else {
                format!("{}", freq)
            };
            draw_text_mut(&mut img, Rgb([0, 0, 0]), 
                15, (y as i32).saturating_sub(8), 
                freq_scale, &font, &label);
            
            for dx in 0..8 {
                if margin_left > dx {
                    img.put_pixel(margin_left - dx, y, Rgb([0, 0, 0]));
                }
            }
        }
    }
    
    draw_text_mut(&mut img, Rgb([0, 0, 0]), 
        10, (margin_top + spec_height / 2 - 40) as i32, 
        axis_scale, &font, "Hz");
    
    draw_db_colorscale(&mut img, margin_left + spec_width + 25, margin_top, 40, spec_height, &font);
    
    img.save(output_path)?;
    Ok(())
}

fn normalize_to_db(spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let max_val = spec.iter()
        .flat_map(|row| row.iter())
        .cloned()
        .fold(0.0f32, f32::max)
        .max(1e-10);
    
    spec.iter()
        .map(|row| {
            row.iter()
                .map(|&val| {
                    if val > 0.0 {
                        20.0 * (val / max_val).log10().max(-80.0)
                    } else {
                        -80.0
                    }
                })
                .collect()
        })
        .collect()
}

fn db_to_color(db: f32) -> (u8, u8, u8) {
    let normalized = ((db + 80.0) / 80.0).clamp(0.0, 1.0);
    
    // Viridis colormap for better visibility
    if normalized < 0.25 {
        let t = normalized / 0.25;
        let r = (68.0 + t * 12.0) as u8;
        let g = (1.0 + t * 38.0) as u8;
        let b = (84.0 + t * 56.0) as u8;
        (r, g, b)
    } else if normalized < 0.5 {
        let t = (normalized - 0.25) / 0.25;
        let r = (80.0 - t * 38.0) as u8;
        let g = (39.0 + t * 107.0) as u8;
        let b = (140.0 - t * 8.0) as u8;
        (r, g, b)
    } else if normalized < 0.75 {
        let t = (normalized - 0.5) / 0.25;
        let r = (42.0 + t * 81.0) as u8;
        let g = (146.0 + t * 52.0) as u8;
        let b = (132.0 - t * 69.0) as u8;
        (r, g, b)
    } else {
        let t = (normalized - 0.75) / 0.25;
        let r = (123.0 + t * 130.0) as u8;
        let g = (198.0 + t * 27.0) as u8;
        let b = (63.0 - t * 18.0) as u8;
        (r, g, b)
    }
}

fn draw_axes(img: &mut RgbImage, x: u32, y: u32, width: u32, height: u32) {
    let black = Rgb([0u8, 0u8, 0u8]);
    
    // Draw thicker axes for better visibility
    for thickness in 0..2 {
        // Left axis
        for dy in 0..=height {
            if x + thickness < img.width() {
                img.put_pixel(x + thickness, y + dy, black);
            }
        }
        
        // Bottom axis
        for dx in 0..=width {
            if y + height + thickness < img.height() {
                img.put_pixel(x + dx, y + height + thickness, black);
            }
        }
    }
}

fn draw_db_colorscale(img: &mut RgbImage, x: u32, y: u32, width: u32, height: u32, font: &Font) {
    // Draw color gradient
    for i in 0..height {
        let db = -80.0 + (80.0 * i as f32 / height as f32);
        let color = db_to_color(db);
        for j in 0..width {
            img.put_pixel(x + j, y + height - 1 - i, Rgb([color.0, color.1, color.2]));
        }
    }
    
    // Draw border
    let black = Rgb([0u8, 0u8, 0u8]);
    for i in 0..height {
        img.put_pixel(x, y + i, black);
        img.put_pixel(x + width - 1, y + i, black);
    }
    for j in 0..width {
        img.put_pixel(x + j, y, black);
        img.put_pixel(x + j, y + height - 1, black);
    }
    
    // Draw dB labels
    let label_scale = Scale::uniform(16.0);
    let db_labels = [0, -20, -40, -60, -80];
    for &db in &db_labels {
        let label_y = y + height - ((db + 80) as f32 / 80.0 * height as f32) as u32;
        draw_text_mut(img, Rgb([0, 0, 0]), 
            (x + width + 8) as i32, (label_y as i32).saturating_sub(8), 
            label_scale, font, &format!("{}", db));
        
        // Draw tick
        for dx in 0..6 {
            if label_y < img.height() {
                img.put_pixel(x + width + dx, label_y, black);
            }
        }
    }
    
    // Draw "dB" label
    let db_label_scale = Scale::uniform(18.0);
    draw_text_mut(img, Rgb([0, 0, 0]), 
        (x + width + 8) as i32, y.saturating_sub(25) as i32, 
        db_label_scale, font, "dB");
}

fn create_mel_filterbank(num_mel_bins: usize, num_fft_bins: usize, sample_rate: u32) -> Vec<Vec<f32>> {
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(sample_rate as f32 / 2.0);
    
    let mel_points: Vec<f32> = (0..=num_mel_bins + 1)
        .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (num_mel_bins + 1) as f32)
        .collect();
    
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<f32> = hz_points.iter()
        .map(|&f| (num_fft_bins as f32 * f / (sample_rate as f32 / 2.0)))
        .collect();
    
    let mut filters = vec![vec![0.0; num_fft_bins]; num_mel_bins];
    
    for i in 0..num_mel_bins {
        let left = bin_points[i];
        let center = bin_points[i + 1];
        let right = bin_points[i + 2];
        
        for j in 0..num_fft_bins {
            let j_f = j as f32;
            if j_f >= left && j_f <= center {
                filters[i][j] = (j_f - left) / (center - left);
            } else if j_f > center && j_f <= right {
                filters[i][j] = (right - j_f) / (right - center);
            }
        }
    }
    
    filters
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

