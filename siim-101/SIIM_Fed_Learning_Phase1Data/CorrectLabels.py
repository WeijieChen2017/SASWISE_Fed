import os
import nibabel as nib
import numpy as np
import sys

def correct_nifti_files(directory):
    # Iterate over every file in the directory
    for file_name in os.listdir(directory):
        # Check if the file is a NIfTI file by its extension
        if file_name.endswith('.nii'):
            input_file_path = os.path.join(directory, file_name)
            output_file_path = input_file_path  # Modify as needed

            # Load the NIfTI file
            nifti_img = nib.load(input_file_path)
            data = nifti_img.get_fdata()

            # Round the data to mitigate floating-point precision issues
            data = np.round(data)

            # Normalize data after rounding
            data_min = data.min()
            data_max = data.max()

            if data_min == data_max:
                # Handle case where all values are the same
                corrected_data = np.zeros(data.shape) if data_min != 1024 else np.ones(data.shape)
            else:
                # Normalize data to have min 0 and max 1, then round again if necessary
                corrected_data = (data - data_min) / (data_max - data_min)
                # Round the normalized data, especially important if dealing with what should be integer values
                corrected_data = np.round(corrected_data)

            # Create a new NIfTI image with the corrected data
            corrected_img = nib.Nifti1Image(corrected_data, affine=nifti_img.affine, header=nifti_img.header)

            # Save the corrected image to disk
            nib.save(corrected_img, output_file_path)

            print(f"Corrected NIfTI file saved to {output_file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <directory_path>")
        sys.exit(1)
    directory_path = sys.argv[1]
    correct_nifti_files(directory_path)
