
from pathlib import Path
from glob import glob
import json
import warnings
import subprocess
import numpy as np
import torch
import SimpleITK as sitk
from scipy.ndimage import label

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
# Image Preprocessing and Resampling
def get_bounding_boxes(ct_sitk, pet_sitk):
    ct_origin = np.array(ct_sitk.GetOrigin())
    pet_origin = np.array(pet_sitk.GetOrigin())
    ct_max = ct_origin + np.array(ct_sitk.GetSize()) * np.array(ct_sitk.GetSpacing())
    pet_max = pet_origin + np.array(pet_sitk.GetSize()) * np.array(pet_sitk.GetSpacing())
    return np.concatenate([np.maximum(ct_origin, pet_origin), np.minimum(ct_max, pet_max)], axis=0)

def resample_images(ct_path, pet_path, resolution=(1, 1, 1)):
    ct = sitk.ReadImage(ct_path, sitk.sitkFloat32)
    pt = sitk.ReadImage(pet_path, sitk.sitkFloat32)
    bb = get_bounding_boxes(ct, pt)
    origin = bb[:3]
    end = bb[3:]
    size = np.round((end - origin) / np.array(resolution)).astype(int)

    def make_resampler(interpolator):
        r = sitk.ResampleImageFilter()
        r.SetOutputSpacing(resolution)
        r.SetOutputOrigin(origin.tolist())
        r.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        r.SetSize(size.tolist())
        r.SetInterpolator(interpolator)
        return r

    ct_resampled = make_resampler(sitk.sitkBSpline).Execute(ct)
    pt_resampled = make_resampler(sitk.sitkBSpline).Execute(pt)
    return ct_resampled, pt_resampled, bb

# ROI cropping
def get_roi_center(pet_tensor, z_top_fraction=0.75, z_score_threshold=1.0):
    crop_z_start = int(z_top_fraction * pet_tensor.shape[2])
    top_of_scan = pet_tensor[..., crop_z_start:]
    mask = ((top_of_scan - top_of_scan.mean()) / (top_of_scan.std() + 1e-8)) > z_score_threshold

    if not mask.any():
        warnings.warn("No high-intensity region found. Using geometric center.")
        center_in_top = np.array(top_of_scan.shape) // 2
    else:
        labeled, num = label(mask.cpu().numpy(), structure=np.ones((3, 3, 3)))
        coords = np.argwhere(labeled == (np.argmax(np.bincount(labeled.ravel())[1:]) + 1)) if num > 0 else np.argwhere(mask.cpu().numpy())
        center_in_top = np.mean(coords, axis=0)

    return (center_in_top + np.array([0, 0, crop_z_start])).astype(int)

def crop_neck_region_sitk(ct_sitk, pet_sitk, crop_box_size=(200, 200, 310), z_top_fraction=0.75, z_score_threshold=1.0):
    pet_np = sitk.GetArrayFromImage(pet_sitk)
    pet_np = np.transpose(pet_np, (2, 1, 0))
    pet_tensor = torch.from_numpy(pet_np).float()

    center = get_roi_center(pet_tensor, z_top_fraction, z_score_threshold)
    box_start = np.clip(center - np.asarray(crop_box_size) // 2, 0, pet_np.shape)
    box_end = np.clip(box_start + crop_box_size, 0, pet_np.shape)
    box_start = np.maximum(box_end - crop_box_size, 0)

    index = box_start.astype(int).tolist()
    size = (box_end - box_start).astype(int).tolist()

    ct_crop = sitk.RegionOfInterest(ct_sitk, size=size, index=index)
    pet_crop = sitk.RegionOfInterest(pet_sitk, size=size, index=index)
    return ct_crop, pet_crop, box_start.tolist(), box_end.tolist()

# Map prediction to original CT space
def prediction_to_original_space(pred_np, meta, box_start_xyz, box_end_xyz, resampled_ct_sitk, original_ct_sitk):
    Zt = box_end_xyz[2] - box_start_xyz[2]
    z_extra = pred_np.shape[0] - Zt
    if z_extra > 0:
        pred_np = pred_np[z_extra // 2:z_extra // 2 + Zt, ...]

    if "roi_start" in meta:
        z0, y0, x0 = meta["roi_start"]
        z1, y1, x1 = meta["roi_end"]
        canvas_cf = np.zeros((Zt, y1 - y0, x1 - x0), dtype=pred_np.dtype)
        canvas_cf[z0:z1, y0:y1, x0:x1] = pred_np
    else:
        canvas_cf = pred_np

    x0, y0, z0 = box_start_xyz
    x1, y1, z1 = box_end_xyz
    canvas_roi = np.zeros(resampled_ct_sitk.GetSize()[::-1], dtype=pred_np.dtype)
    canvas_roi[z0:z1, y0:y1, x0:x1] = canvas_cf

    mask_1mm = sitk.GetImageFromArray(canvas_roi.astype(np.uint8))
    mask_1mm.CopyInformation(resampled_ct_sitk)
    return sitk.Resample(mask_1mm, original_ct_sitk, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
# Utility functions
def load_json_file(*, location):
    with open(location, "r") as f:
        return json.load(f)

def load_image_file_as_array(*, location):
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    if not input_files:
        raise FileNotFoundError(f"No image files found in {location}")
    return input_files[0]

def write_array_as_image_file(*, location, array, filename="output.mha"):
    location.mkdir(parents=True, exist_ok=True)
    if isinstance(array, sitk.Image):
        img = array
    else:
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        img = sitk.GetImageFromArray(array)
    sitk.WriteImage(img, str(location / filename), useCompression=True)
    
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import numpy as np
# === Main run logic ===
def run():
    ct_path = load_image_file_as_array(location=INPUT_PATH / "images/ct")
    pet_path = load_image_file_as_array(location=INPUT_PATH / "images/pet")

    ct_resampled, pet_resampled, bb = resample_images(ct_path, pet_path)
    ct_crop, pet_crop, box_start, box_end = crop_neck_region_sitk(ct_resampled, pet_resampled)

    nii_path = Path("/opt/app/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs")
    result_path = Path("/opt/app/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result")
    nnUNet_results = "/opt/app/nnUNet_results"
    #nii_path.mkdir(parents=True, exist_ok=True)
    #result_path.mkdir(parents=True, exist_ok=True)
    ct_path = Path(ct_path)
    pet_path = Path(pet_path)

    subject_name = ct_path.stem  # 'r45566hgygygy'

    #sitk.WriteImage(ct_crop, nii_path / "subject_0000.nii.gz")
    #sitk.WriteImage(pet_crop, nii_path / "subject_0001.nii.gz")
    sitk.WriteImage(ct_crop, nii_path / f"{subject_name}_0000.nii.gz")
    sitk.WriteImage(pet_crop, nii_path / f"{subject_name}_0001.nii.gz")
    
    print(f"Files in nii_path before prediction: {list(nii_path.glob('*'))}")

    print("Running nnUNetv2_predict...")
    
#    subprocess.run(
#        f"nnUNetv2_predict -i {nii_path} -o {result_path} -d 400 -c 3d_fullres -f all --disable_tta -npp 1 -nps 1",
#        shell=True,
#        check=True
#    )

    # subprocess.run(
    # f"nnUNetv2_predict -i {nii_path} -o {result_path} -d 400 -c 3d_fullres -f all --disable_tta -npp 1 -nps 1 -tr nnUNetTrainerUxLSTMBot",
    # shell=True,
    # check=True
    # )
    # Instantiate predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    # Initialize predictor with your trained model folder and trainer
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset400_Hector_MRI/nnUNetTrainerUxLSTMEnc__nnUNetPlans__3d_fullres'),
        use_folds=('all',),  # Use the folds you want
        checkpoint_name='checkpoint_best.pth',
    )

    # Run prediction from files
    predictor.predict_from_files(
        str(nii_path),
        str(result_path),
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
    

    #seg_path = result_path / "subject.nii.gz"
    seg_path = result_path / f"{subject_name}.nii.gz"
    if not seg_path.exists():
        raise FileNotFoundError(f"Expected prediction not found: {seg_path}")

    seg_crop = sitk.ReadImage(str(seg_path))
    seg_np = sitk.GetArrayFromImage(seg_crop).astype(np.uint8)
    unique_labels = np.unique(seg_np)
    print(f"Unique labels in prediction for {subject_name}: {unique_labels}")
    
    original_ct = sitk.ReadImage(ct_path)
    print(seg_path)
    seg_orig = prediction_to_original_space(
        pred_np=seg_np,
        meta={},
        box_start_xyz=box_start,
        box_end_xyz=box_end,
        resampled_ct_sitk=ct_resampled,
        original_ct_sitk=original_ct,
    )

    # ct_basename = Path(ct_path).stem
    # output_dir = OUTPUT_PATH / "tumor-lymph-node-segmentation"
    # output_dir.mkdir(parents=True, exist_ok=True)

    # write_array_as_image_file(
    #     location=output_dir,
    #     array=seg_orig,
    #     filename=ct_basename + ".mha"
    # )
    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/tumor-lymph-node-segmentation",
        array=seg_orig,
    )
    return 0
def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)
if __name__ == "__main__":
    import sys
    sys.exit(run())
