Installation
============

Prerequisites
-------------

* Python 3.10 or higher
* conda (recommended) or pip

From Source (Development)
-------------------------

1. **Create a conda environment:**

   .. code-block:: bash

      conda create --name costim_screen_env jupyter python=3.12
      conda activate costim_screen_env

2. **Clone or download the repository:**

   .. code-block:: bash

      # If you have the repository as a zip file:
      unzip CostimScreen.zip
      cd CostimScreen

      # Or clone from GitHub (when available):
      # git clone https://github.com/username/CostimScreen.git
      # cd CostimScreen

3. **Install in editable mode:**

   .. code-block:: bash

      pip install -e .

   This installs the package along with all dependencies (numpy, pandas,
   statsmodels, matplotlib, etc.).


From PyPI (Coming Soon)
-----------------------

.. code-block:: bash

   pip install costim-screen


Dependencies
------------

The following packages are automatically installed:

* numpy >= 1.24
* pandas >= 2.0
* statsmodels >= 0.14
* matplotlib >= 3.7
* patsy >= 0.5
* scipy >= 1.10
* openpyxl >= 3.1


Building Documentation
----------------------

To build the documentation locally:

.. code-block:: bash

   pip install sphinx sphinx-autodoc-typehints
   cd docs
   make html

The documentation will be available in ``docs/_build/html/index.html``.