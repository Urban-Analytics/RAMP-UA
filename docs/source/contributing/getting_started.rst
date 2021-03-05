Getting ready to contribute 
===================================

Before starting to contribute we've got this page that runs through all the steps of our contribution process 
and our version control and issue tracking methods. 
Please familiarize yourself with this page before starting to contribute.

Getting in Touch
------------------

Currently, the best way to get in touch is via the `GitHub Issues <https://github.com/Urban-Analytics/RAMP-UA/issues>`_. 

Contributing via GitHub
-------------------------

This project uses `git <https://git-scm.com>`_ for version control and all work is done through `GitHub <https://github.com>`_ as a tool for collaborative working.

In order to contribute via GitHub you'll need to sign up for a free account. 
You can do so by following these `instructions <https://help.github.com/articles/signing-up-for-a-new-github-account/>`_.

Writing in reStructured Text
------------------------------

All RAMP-UA documentation is written using `reStructured text <https://docutils.sourceforge.io/rst.html>`_ syntax and rendered using
the `Sphinx <https://www.sphinx-doc.org/>`__ documentation generator tool.

Follow this `guide <https://sphinx-tutorial.readthedocs.io/step-1/>`_ for a good introduction to using reStructured Text.

Where to start: Issues
------------------------

Issues are a prime way to start contributing to the project. They allow you to create a specific thread related to an issue that can relate to a 
number of subjects highlighted through the available issue tags:

- ``backlog``, a tag for building a backlog of issues todo
- ``bug``, a label to report something that isn't working
- ``documentation``, improvements or additions to documentation
- ``duplicate``, tag for flagging this issue or pull request already exists
- ``enhancement``, a new feature or request 
- ``essential enhancement``, essential to get version 1 of the model working 
- ``help wanted``, extra attention needed from other team members
- ``invalid``, tag for if issue isn't quite right 
- ``not urgent``, tag for issues that don't need to be done urgently
- ``question``, further information requested 
- ``wontfix``, a tag for issues that won't be worked on

Before opening an issue please check carefully a similar issue does not already exist. 
Duplicate issues will be marked and closed. 
You can `open an issue <https://github.com/Urban-Analytics/RAMP-UA/issues/new>`_ now and get contributing.

Making a pull request 
-----------------------

All code development for this project is managed via the Urban Analytics GitHub organisation `RAMP-UA <https://github.com/Urban-Analytics/RAMP-UA/>`__ repository. 
We welcome all contributions through GitHubs `pull request feature <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests>`_
which allows you to make changes on your own separate branch of a repository and suggest them back to our main repository. 
We've broken this down into a couple of steps to help explain how this works:

1. Create an issue or comment on an existing issue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First you need to signal to others working on the repository that you're planning to contribute. 
This helps avoid the duplication of effort and ensures everyone is on the same page about the goal you wish to carry out.
Try to write your issue so that it outlines clearly what you plan to accomplish i.e. 
use the `checklist feature <https://github.blog/2014-04-28-task-lists-in-all-markdown-documents/>`__.

.. _section2-label:

2. Fork the `RAMP-UA <https://github.com/Urban-Analytics/RAMP-UA>`__ repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the fork icon on GitHub to create your own personal copy of the RAMP-UA repository. This means you can tinker to your hearts content without
affecting anyone else.

You'll need to be mindful of keeping `your own fork up to date <https://help.github.com/articles/syncing-a-fork>`_ with the main repository to avoid 
`merge conflicts <https://help.github.com/articles/about-merge-conflicts>`_ between code you've written and code that has been merged 
into the main RAMP-UA repository.

You can then make a local clone of your fork of the repository with the ``git`` commands. ::

    $ git clone https://github.com/your-user-name/RAMP-UA.git RAMP-UA-fork
    $ cd RAMP-UA-fork
    $ git remote add upstream https://github.com/Urban-Analytics/RAMP-UA.git


3. Create a local development environment 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, you'll need to create a local development environment using `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__.
The environment specification is maintained using the provided environment YML file (``environment.yml``) 
from which you can create an identical development environment.

To create this isolated development environment you will need:

* to install either `Anaconda <https://www.anaconda.com/products/individual>`__ or `miniconda <https://docs.conda.io/en/latest/miniconda.html>`__
* to have created a local clone of the repository (as covered in :ref:`section2-label`)
* have ``cd`` into the RAMP-UA-fork directory

We can now use ``conda`` to create the ``ramp-ua`` conda environment which we will use as our isolated development environment 
containing all required dependencies. ::

    # create a new conda environment using the environment.yml specification
    $ conda env create -f environment.yml 
    # enter the ramp-ua environment
    $ conda activate ramp-ua
    # install the package
    $ python setup.py install
    # or install in development mode 
    $ python setup.py develop

4. Make changes as discussed 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Make the changes you've suggested. 
Try and be focused with your work as all pull requests are reviewed and the more changes you make the longer it takes to review them!

Be sure to write good, detailed commit messages.
You can read this `blog <https://chris.beams.io/posts/git-commit/>`_ on how to write good commit messages and see why its important.
We aren't against lots of commits or ones that break the code. 
In fact by default we have a series of tests that run whenever a push is made to a pull request to check if the model still runs 
which is a good way of checking if your changes work.



5. Submit a `pull request <https://help.github.com/articles/creating-a-pull-request>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open a pull request early in your contributing process. 
That helps others on the team see what you're doing and allows others to provide feedback to you in real-time.
When creating a pull request please try and include:

* A description of the problem you're trying to solve (use the # character to reference a specific issue in the Issues section)
* A list of proposed changes to solve the problem 
* Any specifics that reviewers should focus on

If you're still working on your pull request and it isn't ready for a review you can either open a draft pull request or add ``[WIP]`` to
the pull request name.
This signals to others that you're still working on this and it isn't ready for a review just yet.

If during a pull request your changes lead to the continuous integration (CI) tests failing with a notification like "Some checks were not successful"
then the first place to look is by clicking "Details" next to the test which will take you to the GitHub actions page for that test.
From there you can see which step in the actions workflow the code failed at and attempt to troubleshoot.
If you have any questions about this you can raise them on your pull request.
