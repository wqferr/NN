#include <stdlib.h>

#include "core/nn.h"
#include "err.h"

#define ASSERT(expr, err) if (!(expr)) return err;
#define CHECK_NULL(ptr) ASSERT(ptr != NULL, ERR_ILLEGAL_ARG)

typedef struct Layer {
    double *bias;
    double **weights;
} Layer;

/* The neural network does not store the input layer as it does not */
/* need to store any information. */
struct NN {
    /* All hidden layers + output layer */
    Layer *layers;
    /* Number of hidden layers + 1 */
    size_t nLayers;
};

int _nn_layer_create(Layer *, size_t, size_t);
int _nn_destroy_layers(Layer *);

int nn_create(NN **nn, size_t nLayers, const size_t *layerSizes) {
    NN *n;
    int err;
    size_t i, nInputs;

    ASSERT(nLayers > 0, ERR_ILLEGAL_ARG);
    nLayers--;
    
    n = malloc(sizeof(*n));
    CHECK_NULL(n);
    n->nLayers = nLayers;
    n->layers = malloc(nLayers * sizeof(*n->layers));
    CHECK_NULL(n->layers);
    for (i = 0; i < nLayers; i++) {
        ASSERT(layerSizes[i] > 0, ERR_ILLEGAL_ARG);
        err = _nn_layer_create(n->layers + i, layerSizes[i], nInputs);
        if (err)
            return err;
        nInputs = layerSizes[i];
    }

    *nn = n;
    return NO_ERR;
}

int _nn_layer_create(Layer *l, size_t size, size_t numInputs) {
    size_t i;

    l->bias = malloc(size * sizeof(*l->bias));
    CHECK_NULL(l->bias);
    l->weights = malloc(numInputs * sizeof(*l->weights));
    CHECK_NULL(l->weights);
    for (i = 0; i < numInputs; i++) {
        l->weights[i] = malloc(size * sizeof(**l->weights));
        CHECK_NULL(l->weights[i]);
    }
    return NO_ERR;
}
