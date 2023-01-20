from setuptools import setup, find_packages

setup(name='vq_vae',
      version='0.0.1',
      description=('Supplementary code for Hyperbolic Image Embeddings.'),
      url='None',
      author='Valentin Khrulkov, Leyla Mirvakhabova, Evgeniya Ustinova, Victor Lempitsky, Ivan Oseledets',
      author_email='khrulkov.v@gmail.com',
      license='MIT',
      packages=['hyptorch'],
      install_requires=['torch>=0.4.0',
                        'torchvision>=0.2.0',
                        'numpy',
                        'geoopt @ git+https://github.com/geoopt/geoopt.git'],
      zip_safe=False
)
