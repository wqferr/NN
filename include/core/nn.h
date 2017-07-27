
#ifndef NN_H
#define NN_H 1

#include <stddef.h>

typedef struct NN NN;


int nn_create(NN **, size_t, const size_t *);
void nn_destroy(NN *);

#endif
