# Thirdparty Licences

This document lists the licences for software that is either open
source or commercial libraries that are free in the context of our
current use.

With the exception of FMOD Ex 4, we ship no dependencies - you must
obtain and install these yourself. As and when we are able to port our
software to depend on more recent versions of FMOD API (v5), we may
remove the download links for these.

**Note**: Outside of FMOD, we do not distribute any dependencies, as
  they are all runtime requirements only. Their licenses are therefore
  listed as an FYI rather than a condition of redistribution.

  * Python 2.7
    * Python 2.7
    * Licensed under the [Python 2.7 licence](https://www.python.org/download/releases/2.7/license/). Licence text is contained
      in tis directory.
  * Qt 4.8
    * [Qt website](https://www.qt.io/)
    * Copyright notice:
      ```
      The Qt GUI Toolkit is Copyright (C) 2015 The Qt Company Ltd.
      Contact: http://www.qt.io/licensing/
      ```
    * Qt Company Qt LGPL Exception:

      ```
As a special exception to the GNU Lesser General Public License version 2.1, the object code form of a "work that uses the Library" may incorporate material from a header file that is part of the Library. You may distribute such object code under terms of your choice, provided that the incorporated material (i) does not exceed more than 5% of the total size of the Library; and (ii) is limited to numerical parameters, data structure layouts, accessors, macros, inline functions and templates.
      ```
    * Licensed under the LGPL 2.1. Licence text is included in this directory
  * PyQt 4.8
    * [PyQt 4.8](https://www.riverbankcomputing.com/software/pyqt/download)
    * Proprietary, but free for non-commercial use via GPL licence.
    * Licenced under GPL 3, but with the PyQt GPL Exception, which
      allows us to license our own software under the simpler MIT
      license.
    * Copies fo the GPL 3 and PyQt GPL Exception are included in this
      directory.
  * Python modules:
      * [Matplotlib](https://matplotlib.org/index.html)
          * Licensed under the Matplotlib license, a copy of which is in
            this directory.
      * [NumPy](http://www.numpy.org/)
          * BSD 3-clause license, a copy of which is provided in this
            directory.
      * [SciPy](https://www.scipy.org)
          * BSD 3-clause license, a copy of which is provided in this
            directory.
  * FMOD 4.4:
      * [FMOD website](www.fmod.org).
      * Used for stereo sound manipulation and playback in Ticker.
      * Proprietary, but free for non-commercial use.
      * FMOD 4.4 is no longer downloadable from the site, but copies
        of their redistributables are included in our `dependencies`
        directory.
      * We're hoping to port Ticker to use FMOD 5 at some point. If
        you'd like to take a stab at doing so, then there is an
        FMOD 5
        [transitioning guide](http://www.fmod.org/documentation/#content/generated/overview/transitioning.html)
        that may come in handy.
      * We are using it for non-commercial use, and so it is covered
        by their T&C's for non-commercial use. Commercial licensing is
        also available from FMOD.
      * A copy
  * Festival:
      * [Festival website](http://www.cstr.ed.ac.uk/projects/festival/)
      * Used for text-to-speech synthesis in Ticker.
      * Licensed under their X11-type license, a copy of which is
        provided in this directory.
