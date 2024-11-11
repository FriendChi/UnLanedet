import setuptools
import sys
import glob
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

import unlanedet

long_description = "A Tookit for lane detection based on PaddlePaddle"

with open("requirements.txt") as file:
    REQUIRED_PACKAGES = file.read()

def get_extensions():
    extensions = []

    op_files = glob.glob('./unlanedet/layers/ops/csrc/*.c*')
    op_files_ad = glob.glob('./unlanedet/layers/ops/csrc/adnet/*.c*')

    nms_ext_name = 'unlanedet.layers.ops.nms_impl'
    nms_ext_ops = CUDAExtension(
        name=nms_ext_name,
        sources=op_files
    )
    extensions.append(nms_ext_ops)

    # for adnet
    nms_ad_ext_name = 'unlanedet.layers.ops.nms_ad_impl'
    nms_ad_ext_ops = CUDAExtension(
        name=nms_ad_ext_name,
        sources=op_files_ad
    )
    extensions.append(nms_ad_ext_ops)

    return extensions

setuptools.setup(
    name="unlanedet",
    version='0.0.2',
    author="kunyangzhou",
    author_email="zhoukunyangmcgill@163.com",
    description=long_description,
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/zkyseu/PPlanedet",
    packages=setuptools.find_packages(),
    include_package_data=True,
    setup_requires=['cython', 'numpy', 'pytest-runner'],
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    tests_require=['pytest'],
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
    license='MIT License',
    entry_points={'console_scripts': ['pplanedet=pplanedet.command:main', ]})