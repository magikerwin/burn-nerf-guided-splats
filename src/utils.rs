use burn::tensor::{backend::Backend, Tensor, TensorData};

/// Generates a synthetic target image: a red circle centered on a blue background.
pub fn generate_synthetic_target(width: u32, height: u32) -> image::RgbImage {
    let mut img = image::RgbImage::new(width, height);
    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;
    let radius = (width.min(height) as f32) * 0.35;

    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let dx = x as f32 + 0.5 - cx;
        let dy = y as f32 + 0.5 - cy;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < radius {
            // Red circle
            *pixel = image::Rgb([255, 0, 0]);
        } else {
            // Blue background
            *pixel = image::Rgb([0, 0, 128]);
        }
    }
    img
}

/// Converts an RGB image to a normalized Burn Float Tensor of shape [height, width, 3].
pub fn image_to_tensor<B: Backend>(img: &image::RgbImage, device: &B::Device) -> Tensor<B, 3> {
    let (width, height) = img.dimensions();
    let mut data = Vec::with_capacity((width * height * 3) as usize);

    for pixel in img.pixels() {
        data.push(pixel[0] as f32 / 255.0);
        data.push(pixel[1] as f32 / 255.0);
        data.push(pixel[2] as f32 / 255.0);
    }

    let shape = [height as usize, width as usize, 3];
    let tensor_data = TensorData::new(data, shape);
    Tensor::<B, 3>::from_data(tensor_data, device)
}

/// Converts a normalized Burn Float Tensor of shape [height, width, 3] back to an RGB image.
pub fn tensor_to_image<B: Backend>(tensor: Tensor<B, 3>) -> image::RgbImage {
    let shape = tensor.shape();
    let dims = shape.dims::<3>();
    let height = dims[0];
    let width = dims[1];

    // Transfer tensor data from device to host CPU
    let data = tensor.into_data().into_vec::<f32>().expect("Failed to convert tensor to CPU vector");
    let mut img = image::RgbImage::new(width as u32, height as u32);

    let mut data_idx = 0;
    for y in 0..height {
        for x in 0..width {
            let r = (data[data_idx].clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (data[data_idx + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (data[data_idx + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
            data_idx += 3;
        }
    }

    img
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;

    #[test]
    fn test_generate_synthetic_target() {
        let img = generate_synthetic_target(32, 32);
        assert_eq!(img.width(), 32);
        assert_eq!(img.height(), 32);

        // Center pixel should be red [255, 0, 0]
        let center_pixel = img.get_pixel(16, 16);
        assert_eq!(center_pixel, &image::Rgb([255, 0, 0]));

        // Corner pixel should be blue [0, 0, 128]
        let corner_pixel = img.get_pixel(0, 0);
        assert_eq!(corner_pixel, &image::Rgb([0, 0, 128]));
    }

    #[test]
    fn test_image_tensor_conversion() {
        let device = Default::default();
        let img = generate_synthetic_target(16, 16);

        // Convert to tensor
        let tensor = image_to_tensor::<Flex>(&img, &device);
        let shape = tensor.shape();
        let dims = shape.dims::<3>();
        assert_eq!(dims[0], 16);
        assert_eq!(dims[1], 16);
        assert_eq!(dims[2], 3);

        // Convert back to image
        let recon = tensor_to_image::<Flex>(tensor);
        assert_eq!(recon.width(), 16);
        assert_eq!(recon.height(), 16);

        // Check center pixel is preserved
        assert_eq!(recon.get_pixel(8, 8), img.get_pixel(8, 8));
        // Check corner pixel is preserved
        assert_eq!(recon.get_pixel(0, 0), img.get_pixel(0, 0));
    }
}
