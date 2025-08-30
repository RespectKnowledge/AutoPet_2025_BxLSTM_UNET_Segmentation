#  BiXLSTM algorithm for autoPET/CT IV challenge

Source code for the BiXLSTM model container for autoPETIV challenge. Information about the 
submission can be found [here](https://autopet-iv.grand-challenge.org/submission/) 

## Usage 

In order to use this algorithm you need to download weights from https://mega.nz/file/UPVlHIbS#HKZ6qhkWyDmUSMdVj710TWIApiYNn7anuyf9Tre4h0E and put inside fold_all folder.
After that you can build the container by running `bash build.sh`. In order to upload the container, you will need to
save the image via `bash export.sh`.

## Testing

Please check insllation requirements.txt file via `pip install -r requirements.txt`. 
Make sure model weights exist in `/nnUNet_results`. After that you can run `bash test.sh`. 
You can try `bash local_test.sh`

