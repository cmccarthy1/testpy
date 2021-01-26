#define KXVER 3

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#include <stdio.h>
#include "k.h"
#include "Python.h"


EXPORT K k_ptr_to_k(K k) {
    return r1((K)(k->j));
}

