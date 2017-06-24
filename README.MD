# Ticker

This repository contains research implementations of the Ticker and a standard Grid-based single-switch
text-entry systems, and copious amounts of code to run user trials, analyse
results of experiments, and to run information theory/machine learning
simulations of the systems under various noise models.

It is relatively rapidly authored research software, so the following
caveats apply:
  * It has not been extensively QAd, but various versions have been
    used by a few research associates and also trial users. You may
    well run into the odd bug or bit of rusted code, particularly with
    older experiments that I haven't been keeping up-to-date.
  * It is not PEP8 Python. So your linter may have a fit of pique at
    having it's whitespace sensibilities offended.

The primary language is Python, with a smattering of C/C++ bindings to
 allow (a bit of) FMOD to be used from within Python.

The software is licensed under the
[MIT licence](http://www.opensource.org/licenses/MIT), and so you are
free to use it for whatever purpose you choose. Some of our runtime dependencies have more restrictive licences (particularly with regards to commercial use!), so please consult `dependencies/licences/README.MD` for more information.

While information on the library and its use is currently pretty slim,
I'm hoping to link it to as yet unpublished research papers which will
hopefully document the "why" and the "how". In the meantime, consult
the [Directory Structure](#directory_structure) section on where to
find things, including the odd user guide and useful README documents.


# Platforms

  * Ubuntu 14.04: Primary development platform, and so likely to work
    the best. Other Linux platforms should also work well, but you'll
    need to adjust installation instructions.
  * Windows 7: The software has been run on Windows at an earlier
    stage, but this has been left untested for several months, so some
    code rust/rot will likely have set in. Your mileage may vary.

# Installation
## Linux

Steps provided below are for Ubuntu 14.04. They assume you already have a
working Qt 4 installation.

1. Install Python dependencies
   ```sh
   sudo apt-get install python python-qt4 python-matplotlib python-scipy python-numpy festival python-dev pyqt4-dev-tools python-vorbis
   ```

   You may also need to install `python-vorbis`
   ```sh
   sudo apt-get install python-vorbis
   ```

2. Download and unpack the FMOD Ex library.

   Our current implementation uses a somewhat dated version of FMOD (version 4),
   which is no longer directly downloadable from their website (at least not
   without contacting them). For convenience, a copy of their Linux binary
   distribution tarball has been uploaded in subdirectory 'dependencies/fmod_4'

   ```sh
   mkdir some_path/fmod
   cp some_path/ticker/dependencies/fmod_4/fmodapi44464linux.tar.gz some_path/fmod
   cd some_path/fmod
   tar -xzvf fmodapi44464linux.tar.gz
   ln -s fmodapi44464linux fmod
   ```

3. Build Ticker's audio module (C/C++, which uses FMOD)

   We used GCC 4.8 (without C++11 enabled).

   Note that if you already have a suitable version of FMOD on your
   system, then you should not set the `TICKER_FMOD_ROOT_DIR`
   environment variable, and may need to tweak the FMOD include and
   library paths in `setup.py`.

   ```sh
   cd some_path/ticker
   # Tell Ticker's build scripts where you've unpacked FMOD
   # (Note that our stock compilation scripts will expect to find
   # fmod.hpp under the following subdirectory:
   # "$TICKER_FMOD_ROOT_DIR/api/inc/fmod.hpp")
   export TICKER_FMOD_ROOT_DIR="some_path/fmod/fmod"
   # Build using Python's distutils
   python setup.py build
   ```

   This should build a shared library Python module called `audio.so`
   under a platform-specific subdirectory under `ticker/build`, which
   you will need to symlink to be in the top-level Ticker
   directory. E.g.:

   ```sh
   cd  some_path/ticker
   ln -s build/lib*/audio.so
   ```

   Alternatively, add `audio.so` add to your `PYTHONPATH`, e.g. by
   executing the below in the shell, or adding it to your `.bashrc`
   file.

   ```sh
   export PYTHONPATH="$PYTHONPATH":"some_path/ticker/build/lib.linux-x86_64-2.7/audio.so"
   ```

4. Give Ticker a spin!
   ```sh
   cd some_path/ticker
   python ticker.py
   ```

5. Optional: Build the audio unit test. If you've already got Ticker up
   and running, then this step may not be useful. However, if you need
   to debug things related to FMOD audio playback, then the C++ unit test
   is a sensible thing to get running first (so debugging in C++
   only), before tackling the rest of Ticker (debugging across the
   Python/C/C++ language divide).

   ```sh
   cd some_path/ticker/unit_tests/test_ticker_audio
   ./build.sh
   # Run the test
   ./test_alphabet_player
   ```

# Source Overview

## Directory Structure

  * `./`:
      * Contains `ticker.py`, the primary text-entry application of
        this repository and modules related to it.
      * Sadly we don't yet have a user guide for the main
        application, but there are some usage notes on a
        modified Ticker (for running user trials) under
        `./experiments/audio_user_trials/experiment_info` that
        are mostly directly applicable to this application.
  * `./alphabet_sequences/`:
      * Code to optimise the spread or "randomisation" of multiple
        consecutive alphabet sequences so as to minimise the proximity
        of letters to each other. E.g. if the first reading is "a b c
        d", then in the following sequence "a" and "b" should be
        further apart, e.g. "a z q b".
  * `./config`:
      * Configuration files for Ticker applications, specifying the
        alphabet layout in time sequence and stereo position. This
        will be used to play back sounds in the relevant stereo
        position when reading letters to users to chosse
        from. Channel1 means one location in stereo (and one speaker)
        is used. Channel5 means there are 5 stereo locations and
        voices reading the alphabet.
  * `./dependencies`:
      * The FMOD Ex 4 library we depend on isn't directly available to
        most registered FMOD users from the
        [FMOD website](www.fmod.org) anymore. So this directory has
        copies of Windows, Linux and OS X for convenience.
      * This directory also has a subdirectory called `licences` where
        we've aggregated notes on thirdparty software we link against
        at runtime.
  * `./experiments`:
      * This directory contains the largest body of code in the
        repository. Consult `./experiments/README` for more detail on
        the contents, as we only provide a few high-level notes here.
      * Note that the code in the experiments directory may
        require additional dependencies to be installed,
        e.g. pyaudiolab, and python-sckikits-openopt.
      * Code to conduct user trials Ticker and Grid-like
        applications that have been customised to provide target
        phrases to write, which then record user performance in trying
        to write them.
          * Results of the above, stored in anonimised cPickle
            files, and utility code to load and plot the results from
            these.
      * Code to simulate Ticker and Grid-like applications under noisy conditions,
        where you are able to specify the types and levels of noise
        sources.
          * Results of the above, stored in cPickle files, with code
            to load and plot results.
          * _Note that this code is no longer functional after a
            configuration change in another part of the codebase, but
            I hope to fix this soon!_
      * A Grid-like text entry implementation in `grid.py`. Having this text
        entry application buried away here is a bit odd, so it is likely that
        it will be moved to sit alongside `ticker.py` in the top-level
        directory in due course.
      * Cleaned-up recordings of alphabet letters read by various
        people (used to prompt single-switch users to choose letters),
        as well as a few notes on how to go about recording and
        processing letter recordings so that they are suitable for an
        HMI interface (not noisy, right lengths, etc).
      * Useful PDF documents outlining user-guide type information for
        trial users, can be found here:
            * `experiments/audio_user_trials/experiment_info`
            * `experiments/multi_channel_user_trials/experiment_info`


# Dependencies

The following is a non-exhaustive list of dependencies used. See
`dependencies/licences/README.MD` for more information.

  * Python 2.7
    * Python 3 is unlikely to work out of the box, as there are some
      Python C modules that will need to be updated.
  * Qt 4.8
    * More recent, and potentially slightly older Qt 4 versions should
      work fine.
    * No attempt has been made to port things to Qt 5. Given the heavy
      dependency on PyQt, you will need to cross-check the level of
      expected support provided for that package against Qt 5.
  * PyQt 4.8
  * Python modules:
      * matplotlib
      * numpy
      * scipy
      * festival
  * FMOD 4.4:
      * FMOD 4.4 is no longer downloadable from the
        [FMOD website](www.fmod.org).
      * For convenience our repository contains some of their
        distributables (headers and library files) under the
        `dependencies` directory.
      * If you'd rather get these from FMOD themselves, then their
        support team may well be persuaded to make a download
        available for you if you ask very nicely.
      * I'm hoping to port Ticker to use FMOD 5 at some point. If
        you'd like to take a stab at doing so, then there is a handy
        FMOD 5
        [transitioning guide](http://www.fmod.org/documentation/#content/generated/overview/transitioning.html)
        that may come in handy.
  * Festival:
      * [Festival website](https://www.qt.io/)
      * Licensed under their X11-type license.

# Miscellaneous Notes

## Useful Linux audio-related commandlines

```sh
sudo /etc/init.d/alsa-utils restart
killall pulseaudio
oggenc -o asterisk-caller.ogg prompt03.wav
soundconverter uvwxyz.wav tmp.ogg
```

## Ticker settings (tiny subset of)

1. Free writing: python ticker.py in ticker-0.0.0
   Undo (after receiving clicks):
   1. More than 4 or more clicks (set number at View -> Settings ->
      Undo clicks), will repeat current letter (if you know you made
      an erroneous click)
   2. At the end of a sequence the 3 most likely letters will be read
      back to the user.  A 2 second period will follow: if the user
      clicks, the user will get a chance to do some more clicking.
2. Tutorial program: python ticker_demo.py in ticker-0.0.0 (without
   undo functions)
3. Remove animation: View -> Alphabet (close dock widget at the
   bottom)
4. Change dictionary: Edit-> Dictionary (select a dictionary in text
   file, there are two in the "dictionaries" directory
5. Change initial click distribution: Settings.

## Code generation from Qt UI layout files

There are a few Qt Designer (`designer`) `.ui` files to generate some
of the UI. If you tweak them, you'll need to regenerate some code:

```sh
pyuic4 volume_layout.ui -o volume_editor_layout.py
```

There is a packaged script that should generate the Python files for
most of the .ui files: `./script_files/build_gui_layout.sh`

# Authors

  * Primary author: Emli-Mari Nel
  * Special thanks to Patrick Welche for his help with the Windows port.
