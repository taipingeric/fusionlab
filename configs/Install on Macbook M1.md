#Install on Macbook M1 chip

Ref: [https://github.com/jeffheaton/t81_558_deep_learning/install/tensorflow-install-conda-mac-metal-jan-2023.ipynb](https://github.com/jeffheaton/t81_558_deep_learning/install/tensorflow-install-conda-mac-metal-jan-2023.ipynb)

[video link](https://www.youtube.com/watch?v=5DgWvU0p2bk)

**NOTE: It's been tested on Macbook Air 2021 M1 chip only**

Requirements:
* Apple Mac with M1 chips
* MacOS > 12.6 (Monterey)

Following steps
1. Clone this repo
```bash
git clone https://github.com/taipingeric/fusionlab.git
cd fusionlab
```
2. Uninstall Anaconda (Optional): https://docs.anaconda.com/anaconda/install/uninstall/
3. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   1. Miniconda3 macOS Apple M1 64-bit pkg
   2. Miniconda3 macOS Apple M1 64-bit bash
4. Install the xcode-select command-line
```bash
xcode-select --install
```
5. Deactivate the base environment
```bash
conda deactivate 
```
7. Clone this repo from github and change dir
```bash
git clone https://github.com/taipingeric/fusionlab.git
cd fusionlab
```
8. Create conda environment using [config](./tf-apple-m1-conda.yaml)
```bash
conda env create -f ./configs/tf-apple-m1-conda.yaml -n fusionlab
```
7. Replace [requirements.txt](../requirements.txt) with [requirements-m1.txt](requirements-m1.txt)
8. Install by pip
```bash
pip install -r requirements-m1.txt
```
