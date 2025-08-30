#import json
#import os
#import shutil
#import subprocess
#from pathlib import Path
#import SimpleITK
#import torch
#
#from utils import save_click_heatmaps

#class Autopet_baseline:
#
#    def __init__(self):
#        """
#        Write your own input validators here
#        Initialize your model etc.
#        """
#        # set some paths and parameters
#        # according to the specified grand-challenge interfaces
#        self.input_path = "/input/"
#        # according to the specified grand-challenge interfaces
#        self.output_path = "/output/images/tumor-lesion-segmentation/"
#        self.nii_path = (
#            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs"
#        )
#        self.lesion_click_path = (
#            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/clicksTs"
#        )
#        self.result_path = (
#            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result"
#        )
#        self.nii_seg_file = "TCIA_001.nii.gz"
#        pass
#
#    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  # nnUNet specific
#        img = SimpleITK.ReadImage(mha_input_path)
#        SimpleITK.WriteImage(img, nii_out_path, True)
#
#    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  # nnUNet specific
#        img = SimpleITK.ReadImage(nii_input_path)
#        SimpleITK.WriteImage(img, mha_out_path, True)
#    
#    def gc_to_swfastedit_format(self, gc_json_path, swfast_json_path):
#        with open(gc_json_path, 'r') as f:
#            gc_dict = json.load(f)
#        swfast_dict = {
#            "tumor": [],
#            "background": []
#        }
#        
#        for point in gc_dict.get("points", []):
#            if point["name"] == "tumor":
#                swfast_dict["tumor"].append(point["point"])
#            elif point["name"] == "background":
#                swfast_dict["background"].append(point["point"])
#        with open(swfast_json_path, 'w') as f:
#            json.dump(swfast_dict, f)
#
#    def check_gpu(self):
#        """
#        Check if GPU is available
#        """
#        print("Checking GPU availability")
#        is_available = torch.cuda.is_available()
#        print("Available: " + str(is_available))
#        print(f"Device count: {torch.cuda.device_count()}")
#        if is_available:
#            print(f"Current device: {torch.cuda.current_device()}")
#            print("Device name: " + torch.cuda.get_device_name(0))
#            print(
#                "Device memory: "
#                + str(torch.cuda.get_device_properties(0).total_memory)
#            )
#
#    def load_inputs(self):
#        """
#        Read from /input/
#        Check https://grand-challenge.org/algorithms/interfaces/
#        """
#        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))[0]
#        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))[0]
#        uuid = os.path.splitext(ct_mha)[0]
#
#        self.convert_mha_to_nii(
#            os.path.join(self.input_path, "images/ct/", ct_mha),
#            os.path.join(self.nii_path, "TCIA_001_0000.nii.gz"),
#        )
#        self.convert_mha_to_nii(
#            os.path.join(self.input_path, "images/pet/", pet_mha),
#            os.path.join(self.nii_path, "TCIA_001_0001.nii.gz"),
#        )
#        
#        json_file = os.path.join(self.input_path, "lesion-clicks.json")
#        print(f"json_file: {json_file}")
#        self.gc_to_swfastedit_format(json_file, os.path.join(self.lesion_click_path, "TCIA_001_clicks.json"))
#
#        click_file = os.listdir(self.lesion_click_path)[0]
#        if click_file:
#            with open(os.path.join(self.lesion_click_path, click_file), 'r') as f:
#                clicks = json.load(f)
#            save_click_heatmaps(clicks, self.nii_path, 
#                                os.path.join(self.nii_path, "TCIA_001_0001.nii.gz"),
#                                )
#        print(os.listdir(self.nii_path))
#
#        return uuid
#
#    def write_outputs(self, uuid):
#        """
#        Write to /output/
#        Check https://grand-challenge.org/algorithms/interfaces/
#        """
#        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
#        self.convert_nii_to_mha(
#            os.path.join(self.result_path, self.nii_seg_file),
#            os.path.join(self.output_path, uuid + ".mha"),
#        )
#        print("Output written to: " + os.path.join(self.output_path, uuid + ".mha"))
#
#    def predict(self):
#        """
#        Your algorithm goes here
#        """
#        print("nnUNet segmentation starting!")
#        cproc = subprocess.run(
#            f"nnUNetv2_predict -i {self.nii_path} -o {self.result_path} -d 221 -c 3d_fullres -f 0 --disable_tta",
#            shell=True,
#            check=True,
#        )
#        print(cproc)
#        # since nnUNet_predict call is split into prediction and postprocess, a pre-mature exit code is received but
#        # segmentation file not yet written. This hack ensures that all spawned subprocesses are finished before being
#        # printed.
#        print("Prediction finished")
#
#   
#    def process(self):
#        """
#        Read inputs from /input, process with your algorithm and write to /output
#        """
#        # process function will be called once for each test sample
#        self.check_gpu()
#        print("Start processing")
#        uuid = self.load_inputs()
#        print("Start prediction")
#        self.predict()
#        print("Start output writing")
#        self.write_outputs(uuid)
#
#
#if __name__ == "__main__":
#    print("START")
#    Autopet_baseline().process()
    
####################################################################

import json
import os
import shutil
import subprocess
from pathlib import Path
import SimpleITK
import torch
import numpy as np

#from utils import save_click_heatmaps
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Autopet_baseline:

    def __init__(self):
        """
        Initialize paths and parameters
        """
        self.input_path = "/input/"
        self.output_path = "/output/images/tumor-lesion-segmentation/"
        self.nii_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs"
        )
        self.result_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result"
        )
        self.nii_seg_file = "TCIA_001.nii.gz"

        # nnUNet results base path (update if needed)
        self.nnUNet_results = "/opt/algorithm/nnUNet_results"

    # -------------------------------
    # Data conversion utilities
    # -------------------------------
    def convert_mha_to_nii(self, mha_input_path, nii_out_path):
        """Convert .mha ? .nii.gz"""
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):
        """Convert .nii.gz ? .mha"""
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    # -------------------------------
    # GPU check
    # -------------------------------
    def check_gpu(self):
        """Check if GPU is available"""
        print("Checking GPU availability")
        is_available = torch.cuda.is_available()
        print("Available:", is_available)
        print(f"Device count: {torch.cuda.device_count()}")
        if is_available:
            dev_id = torch.cuda.current_device()
            print(f"Current device: {dev_id}")
            print("Device name:", torch.cuda.get_device_name(dev_id))
            print("Device memory:", torch.cuda.get_device_properties(dev_id).total_memory)

    # -------------------------------
    # Input preparation (CT + PET only)
    # -------------------------------
    def load_inputs(self):
        """
        Read input data, convert CT + PET MHA to NIfTI
        """
        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))[0]
        uuid = os.path.splitext(ct_mha)[0]

        # Convert CT ? channel 0
        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/ct/", ct_mha),
            os.path.join(self.nii_path, "TCIA_001_0000.nii.gz"),
        )

        # Convert PET ? channel 1
        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/pet/", pet_mha),
            os.path.join(self.nii_path, "TCIA_001_0001.nii.gz"),
        )

        print("Prepared input NIfTI files:", os.listdir(self.nii_path))
        return uuid

    # -------------------------------
    # Save outputs
    # -------------------------------
    def write_outputs(self, uuid):
        """
        Write results to /output/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(
            os.path.join(self.result_path, self.nii_seg_file),
            os.path.join(self.output_path, uuid + ".mha"),
        )
        print("Output written to:", os.path.join(self.output_path, uuid + ".mha"))

    # -------------------------------
    # nnUNetv2 prediction
    # -------------------------------
    def predict(self):
        """
        Run nnUNetv2 prediction
        """
        print("nnUNet segmentation starting!")

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            #device=torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"),
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )

        # Initialize predictor with your trained model folder
        model_folder = join(
            self.nnUNet_results,
            "Dataset222_AutoPetTask1/nnUNetTrainerUxLSTMBot__nnUNetPlans__3d_fullres",
        )
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=("all",),  # or (0,) for single fold
            checkpoint_name="checkpoint_best.pth",
        )

        predictor.predict_from_files(
            str(self.nii_path),
            str(self.result_path),
            save_probabilities=False,   # set True if you want prob maps
            overwrite=False,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )
        
#        # Optional: print GPU memory after prediction
#        if torch.cuda.is_available():
#            print(f"GPU memory after prediction: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")

        print("Prediction finished")

    # -------------------------------
    # Entrypoint
    # -------------------------------
    def process(self):
        """Main entrypoint"""
        self.check_gpu()
        print("Start processing")
        uuid = self.load_inputs()
        print("Start prediction")
        self.predict()
        print("Start output writing")
        self.write_outputs(uuid)


if __name__ == "__main__":
    print("START")
    Autopet_baseline().process()

   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
