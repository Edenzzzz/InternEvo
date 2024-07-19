import os
import platform
import subprocess
import sys

from setuptools import find_packages, setup

# package info
PACKAGE_NAME = "InternEvo"
PACKAGE_DESCRIPTION = "An open-source, lightweight framework designed to facilitate model training without requiring extensive dependencies."

# package option
FLASH_ATTN_VERSION = "2.3.1.post1"
TORCH_VERSION = "2.1"
CXX11ABI_FLAG = False

CUDA_HOME = os.environ.get("CUDA_HOME")
if not CUDA_HOME:
    CUDA_HOME = os.environ.get("CUDA_PATH")
if not CUDA_HOME:
    raise RuntimeError('Environment variable CUDA_HOME or CUDA_PATH should be set.')

pwd = os.path.dirname(__file__)

def get_cuda_bare_metal_version(cuda_home):
    output = subprocess.check_output([cuda_home + "/bin/nvcc", "-V"], universal_newlines=True).split()
    release_idx = output.index("release")
    return output[release_idx+1].split(",")[0].replace('.', '')

def get_version():
    with open(os.path.join(pwd, "version.txt"), "r") as f:
        content = f.read()
    return content


def get_long_description():
    with open(os.path.join(pwd, "README.md")) as f:
        content = f.read()
    return content

def parse_requirements(filename):
    with open(filename, "r") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#") and "torch-scatter" not in line
        ]
    return lines

def main():
    # check env
    assert sys.platform.startswith("linux"), "InternEvo cuurently only works for linux"
    platform_name = f"{sys.platform}_{platform.uname().machine}"
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    cuda_version = get_cuda_bare_metal_version(CUDA_HOME)
    cxx11abi_flag = str(CXX11ABI_FLAG).upper()

    # Initialize lists for dependencies
    install_requires = []
    dependency_links = []
    extra_index_url = []

    # Parse each .txt file and extract requirements
    requirements_folder = "requirements"
    requirements_files = [
        os.path.join(requirements_folder, f) for f in os.listdir(requirements_folder) if f.endswith(".txt")
    ]
    for requirements_file in requirements_files:
        requirements = parse_requirements(requirements_file)
        for req in requirements:
            if req.startswith("-f "):
                # Handle -f clauses
                url = req.split(" ")[1].strip()
                dependency_links.append(url)
            elif req.startswith("--extra-index-url"):
                # Handle --extra-index-url
                url = req.split(" ")[1].strip()
                extra_index_url.append(url)
            else:
                # Standard package requirement
                install_requires.append(req)
    if "torch" in install_requires:
        extra_index_url.append(f"https://download.pytorch.org/whl/cu{cuda_version}")

    # add rotary_emb and xentropy wheels
    install_requires.extend(["rotary_emb", "xentropy"])

    # add flash-attn wheels
    install_requires.append(
        f"flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v{FLASH_ATTN_VERSION}/flash_attn-{FLASH_ATTN_VERSION}+cu{cuda_version}torch{TORCH_VERSION}cxx11abi{cxx11abi_flag}-{python_version}-{python_version}-{platform_name}.whl#egg=flash-attn>={FLASH_ATTN_VERSION}"
    )

    setup(
        name=PACKAGE_NAME,
        version=get_version(),
        description=PACKAGE_DESCRIPTION,
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        url="https://github.com/InternLM/InternEvo",
        project_urls={
            "Bug Tracker": "https://github.com/InternLM/InternEvo/issues",
        },
        license="Apache License 2.0",
        packages=find_packages(),
        python_requires=">=3.10",
        install_requires=install_requires,
        dependency_links=dependency_links,
        extra_index_url=extra_index_url,
        classifiers=[
            "Programming Language :: Python :: 3.10",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: Unix",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
        ],
    )

if __name__ == "__main__":
    main()
