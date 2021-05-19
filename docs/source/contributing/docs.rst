Contributing to the documentation
===================================

Documentation is **super** important and making sure we have straightforward, up to date 
documentation is a crucial part of making sure this project is as reproducible as possible.

This project has been put together by a number of different collaborators so there may be bits of 
the code you find documented well and sections in need of improvement. 
We're aware of this and are trying our best to improve the documentation. But you can help!

*Before starting on this page make sure you've read our* :doc:`get ready to contribute guide <getting_started>`.

About RAMP-UA Documentation
------------------------------

Documentation for this package is generated using `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`__ and 
is written in `reStructured text <https://docutils.sourceforge.io/rst.html>`_. 

Documentation is found in two primary places: documentation pages like this written by hand in ``docs/source`` and docstrings that are found 
within code functions and classes. `Docstrings <https://www.python.org/dev/peps/pep-0257/>`__ aim to provide a specific overview of the behaviour of the code component it is within whilst documentation pages 
are more general guides for the project as a whole.

Building documentation locally
----------------------------------

You can build and test documentation as you contribute to it using the following instructions when you are within the ``ramp-ua`` conda environment.
To generate the specific API reference documentation from docstrings we use the ``sphinx.ext.autodoc`` extension which automatically extracts 
docstrings and renders them in the documentation pages. Using this extension is specified within the ``docs/source/conf.py`` file so that sphinx knows to use it.

To build the documentation locally: ::

    $ cd docs/

    # remove any previous rendered documentation
    $ make clean

    # create new html documentation
    $ make html


If a new module is added you will need to create new `.rst` files using the ``sphinx-apidoc`` command. ::

    $ cd docs/

    $ sphinx-apidoc -f -o source/ ../new_module/


This will generate new ``.rst`` files from the new modules docstrings that can then be rendered into html by running ``make html``.
