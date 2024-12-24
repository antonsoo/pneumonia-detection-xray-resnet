How to download the data:

1) Install the Kaggle API:

```bash
pip install kaggle
```

2) Set up Kaggle API Credentials:

-  Create a Kaggle Account: If you don't have one, sign up at https://www.kaggle.com/.
-  Create an API Token:
  -  Go to your Kaggle account page (click on your profile picture in the top right corner, then select "Account").
  -  Scroll down to the "API" section and click "Create New API Token."
  -  This will download a file called kaggle.json to your computer.
-  Place kaggle.json:
  - Move the kaggle.json file to the ~/.kaggle/ directory (create it if it doesn't exist). On Linux/macOS, this is a hidden directory in your home folder. On Windows, it's usually C:\Users\<your_username>\.kaggle\.
  - Make sure the file has the correct permissions: chmod 600 ~/.kaggle/kaggle.json (on Linux/macOS).

3) Save `download_data.py`: Save the Python code above as a file named `download_data.py` in your project's src directory.

4) Run the Script:
-  Open your terminal or command prompt.
-  Navigate to your project's src directory:

```bash
cd pneumonia-detection-xray-resnet/src
```

- Run the script with the dataset name and destination folder as arguments:

```bash
python download_data.py paultimothymooney/chest-xray-pneumonia data/chest_xray
```

This will download the dataset to the `data/chest_xray` folder within your project.
