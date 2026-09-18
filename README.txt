BITS F445 - Programming Assignment 1
Supporting Files

Dataset:
Each row represents one simulated day of operation of a building.
The target is daily electricity consumption in kWh.

The input features x1, x2, ..., x100 are anonymized sensor/operational measurements.

Files:

* train.csv
* validation.csv
* GRP000_2027A7PS0000.json   (sample model JSON file)
* check_model_json.py        (JSON model checker)

Checking the sample JSON:
Keep all the above files in the same directory and run:

python3 check_model_json.py GRP000_2027A7PS0000.json

The checker should report whether the JSON file is valid and display its validation MSE.

Note:
The dataset is synthetic, and the feature names are intentionally anonymized.

