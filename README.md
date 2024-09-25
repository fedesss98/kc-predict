# Kc predictions via ETa ML models
*A. Pagano, F. Amato, M. Ippolito, D. De Caro, D. Croce*

Here lies the code to predict Evapotranspiration (ETa) and thus the Crop-Coefficient (Kc) of a given crop using Machine Learning (ML) models. 
This code produces predictions of ETa .

**The most updated version of this code is currently in development in the branch** `main`.

Data pipeline is as follows:
- **Preprocessing Data**
    - Scaling
    - Missing measures Imputation
- **Training Models**
- **ETa Prediction**
- **Postprocessing**
    - **Seasonal Decomposition**
    - **Filtering**
    - **Evaluation**

---
## How to use this code
First things first, you need to clone this repository and pull changes from `better-interface` branch.

Start creating a new folder where you want to place this code.
Install Git in your machine and clone this repository in that directory:
```git
git clone https://github.com/fedesss98/kc-predict.git
```
The created `kc-predict` directory will be your **Root Folder**.
Now create a new local brach and checkout into it, 
```git
git checkout -b better-interface
```
Then fetch new commits from the remote repository
```git
git fetch
```
and pull changes in the current local `better-interface` branch,
```git
git pull origin better-interface
```
\
You will need Python along with different libraries to run this code: the easiest way is to set up or update a Conda Environment using the file `environment.yml`.
```
$conda env create --file environment.yml
```
By default, the environment is named `ml`, but you can change it in the `environment.yml` file.
Activate the environment with
```
$conda activate ml
```
Now you're ready to run this code.

All the setting up for this workflow happens in the file `config.toml`, and the program is executed running the file `run.bat`---both located in the Root Directory---without messing with the actual code. 
Alternatively, you can run the main script `kcpredict/kcpredict.py`. 
All you need to do as a user is setting up your configuration file, then all the pipeline is run from start to finish, figures are shown and the output is saved.

### Setting the Configuration
Configuration file is written in TOML format which make it easily readable. 
Settings are divided according to the data pipeline shown before. 
It is all based on an input/output logic that consent to personalize project folders structure. 
Each step of the pipeline reads data in an input folder and save processed data in an output folder. 
An example of a configuration file is given, named `example_config.toml`. 
Start making a copy of it and renaming it `config.toml` before changing parameters. 

#### Configure the Project
Every run of this program is to be considered a new instance of a project. Every project should have its directory. 
There are two parameters to set up a new project: `id` and `project_dir`. 
The **ID** is a wildcard you can use to give a name or identify a specific run, but it's never used internally. 
The **Project Directory** configure the root folder for the run: all paths for inputs and outputs are specified from that location, except for the position of the initial input. 
This is not to be confused with the **Root Folder** which is the place where all this code lies.

**Project Directory** should be an absolute path like
`C:\users\fedesss\kcpredict\my-new-run` or a relative path from the Root Directory like `.\my-new-run`. 
The program creates this directory if it does not exist, along with all other directories specified in the configuration file. Moreover, a copy of the configuration file is pasted in this folder. Also, data are copied from their location to this Project Directory.

#### Configure Dataset
To set up the initial creation of the raw dataset, use the tag `[make-data]`. You can set:

- **`input_file`** specify the name and location of data file (CSV or Excel or Pickle) *relative to the Root Folder* .
- **`output_path`** specify the location *relative to the Project Directory* where to save the raw data DataFrame in pickle format, under the name of `data.pickle`.
- **`visualize`** is a flag used to show or not the raw data series.

#### Preprocess Data
Use the `[preprocess]` tag to set preprocessing options. Here data are scaled and imputed, and separated in a *train/test* dataset and a *prediction* dataset, where there are no measures for the target quantity to predict. Also, *test/train* data are split into **folds** to cross-validate each model on them.

- **`input_path`** specify the relative location of the file `data.pickle`;
- **`output_path`** specify the location of the test dataset and all the training folds;
- **`features`** lists all the feature names as they appear in the CSV or Excel file;
- **`scaler`** can be `"Standard"` or `"MinMax"` based on the choice of scaling;
- **`folds`** is the number of equal folds in which data is split
- **`k_seed`** is the initial seed for the K-folding algorithm
- **`visualize`** is a flag used to show or not the preprocessed data series.


