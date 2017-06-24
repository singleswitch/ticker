#include <Python.h>

#include "audio.h"

static PyObject *
audio_playNext(PyObject *self, PyObject *args)
{
	if (!PyArg_ParseTuple(args, ""))
		return NULL;

	printf("audio_playNext() was called\n");
	playNext();

	return Py_None;
}

static PyMethodDef audio_methods[] = {
	{"playNext", audio_playNext, METH_VARARGS,
		"Start playing the next sound"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initaudio(void)
{
	PyObject *m;

	m = Py_InitModule("audio", audio_methods);
	if (m == NULL)
		return;
}
