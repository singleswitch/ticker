#include "audio.h"
#include <Python.h>

static PyObject *audio_playNext(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    playNext();
    return Py_None;
}

static PyObject *audio_isReady(PyObject *self, PyObject *args)
{
    int is_ready;
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    is_ready = isReady();
    return Py_BuildValue("i", is_ready);
}

static PyObject *audio_setChannels(PyObject *self, PyObject *args)
{
    int nchannels;
    if (!PyArg_ParseTuple(args, "i", &nchannels))
        return NULL;
    setChannels(nchannels);
    return Py_None;
}

static PyObject *audio_setTicks(PyObject *self, PyObject *args)
{
    int nticks;
    if (!PyArg_ParseTuple(args, "i", &nticks))
        return NULL;
    setTicks((unsigned int)nticks);
    return Py_None;
}

static PyObject *audio_restart(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    restart();
    return Py_None;
}

static PyObject *audio_stop(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    stop();
    return Py_None;
}

static PyObject *audio_playInstruction(PyObject *self, PyObject *args)
{
    char *instruction;
    char *file_type;
    if (!PyArg_ParseTuple(args, "ss", &instruction, &file_type))
        return NULL;
    playInstruction(instruction, file_type);
    return Py_None;
}

static PyObject *audio_setRootDir(PyObject *self, PyObject *args)
{
    char *dir_name;
    if (!PyArg_ParseTuple(args, "s", &dir_name))
        return NULL;
    setRootDir(dir_name);
    return Py_None;
}

static PyObject *audio_setAlphabetDir(PyObject *self, PyObject *args)
{
    char *dir_name;
    if (!PyArg_ParseTuple(args, "s", &dir_name))
        return NULL;
    setAlphabetDir(dir_name);
    return Py_None;
}

static PyObject *audio_setConfigDir(PyObject *self, PyObject *args)
{
    char *dir_name;
    if (!PyArg_ParseTuple(args, "s", &dir_name))
        return NULL;
    setConfigDir(dir_name);
    return Py_None;
}

static PyObject *audio_setVolume(PyObject *self, PyObject *args)
{
    float volume;
    int channel;
    if (!PyArg_ParseTuple(args, "fi", &volume, &channel))
    {
        return NULL;
    }
    setVolume(volume, channel);
    return Py_None;
}

static PyObject *audio_isPlayingInstruction(PyObject *self, PyObject *args)
{
    int is_playing;
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    is_playing = isPlayingInstruction();
    return Py_BuildValue("i", is_playing);
}

static PyObject *audio_getCurLetterTimes(PyObject *self, PyObject *args)
{
    int size, n, ok;
    PyObject *o_list;
    const int *letter_times;
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    size = 0;
    ok = 0;
    n = 0;
    letter_times = getCurLetterTimes(&size);
    o_list = PyList_New(0);
    for (; n < size; ++n)
    {
        ok = PyList_Append(o_list, Py_BuildValue("i", *letter_times++));
    }
    return o_list;
}

static PyMethodDef audio_methods[] = {
    {"playNext", audio_playNext, METH_VARARGS, "Start playing the next sound"},
    {"isReady", audio_isReady, METH_VARARGS, "Is ready for next sound"},
    {"setChannels", audio_setChannels, METH_VARARGS, "Set number of channels"},
    {"restart", audio_restart, METH_VARARGS, "Restart audio"},
    {"stop", audio_stop, METH_VARARGS, "Stop audio"},
    {"playInstruction", audio_playInstruction, METH_VARARGS,
     "Playing back a prerecorded instruction (tried .ogg or .wav sound file)."},
    {"setRootDir", audio_setRootDir, METH_VARARGS,
     "Set the root directory of audio command files ."},
    {"setAlphabetDir", audio_setAlphabetDir, METH_VARARGS,
     "Set the alphabet sound-file directory."},
    {"setConfigDir", audio_setConfigDir, METH_VARARGS,
     "Set the config sound-file directory."},
    {"setVolume", audio_setVolume, METH_VARARGS, "Set channel volume"},
    {"isPlayingInstruction", audio_isPlayingInstruction, METH_VARARGS,
     "Is busy playing an instruction"},
    {"getCurLetterTimes", audio_getCurLetterTimes, METH_VARARGS,
     "Get current positions in sound files: -1 if sound is not playing"},
    {"setTicks", audio_setTicks, METH_VARARGS,
     "Set number of ticks before alphabet plays"},
    {NULL, NULL, 0, NULL}};

PyMODINIT_FUNC initaudio(void)
{
    PyObject *m;

    m = Py_InitModule("audio", audio_methods);
    if (m == NULL)
        return;
}
