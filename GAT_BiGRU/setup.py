from setuptools import setup
from setuptools import find_packages

setup(name='GAT-BiGRU',
      version='1',
      description='Spatiotemporal Joint Network, GAT extracts spatial information, BiGRU extracts temporal information, and finally performs classification prediction',
      author='Shi,Hongtao and Yang,Fengdong',
      author_email='sht@qau.edu.cn,namezhuge@outlook.com',
      download_url='https://github.com/name-zhuge/YOLOv8-SLS-and-GAT-BiGRU',
      license='QingDao Agriculture University',
      install_requires=['tensorflow==2.15.0',
			    'pytorch==2.3.1',
			    'pandas==2.2.3',
			    'numpy==1.26.4',
			    'scikit-learn==1.5.2',
			    'matplotlib==3.9.2',
			    'scipy==1.13.1'
                        ],
      packages=find_packages())