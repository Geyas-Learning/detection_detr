from PIL import Image
import numpy as np
import os
import argparse


def apply_threshold(channel):
    # apply thresholding to an image channel
    threshold_value = channel.getextrema()[1] * 0.5
    return channel.point(lambda x: True if x > threshold_value else False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert image files to NPZ files')
    parser.add_argument('--input_dir', type=str, help='Path to the folder containing mask JPG files')
    parser.add_argument('--output_dir', type=str, help='Path to the folder to save NPZ files')

    args = parser.parse_args()

    input_folder = args.input_dir
    output_folder = args.output_dir

    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List PNG files in the input folder
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

    for png_file in png_files:
        # Open the PNG image
        img = Image.open(os.path.join(input_folder, png_file))

        # Split the image into channels
        channels = img.split()

        # Apply thresholding to each channel
        thresholded_channels = [apply_threshold(channel) for channel in channels]

        # Merge the thresholded channels into a single image
        thresholded_image = Image.merge('RGB', thresholded_channels)

        # Convert the thresholded image to a numpy array
        thresholded_np_array = np.array(thresholded_image, dtype=np.bool_)

        # Save the numpy array to a NPZ file
        output_filename = os.path.splitext(png_file)[0] + '.npz'
        output_path = os.path.join(output_folder, output_filename)
        np.savez_compressed(output_path, data=thresholded_np_array)
