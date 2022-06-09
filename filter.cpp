

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

struct filter_state;

struct filter_state *filter_state_create(void);
void filter_state_destroy(struct filter_state *pObject);
void filter_state_reset(struct filter_state *pThis);
int filter_apply(struct filter_state *pThis, float *pInput, float *pOutput,
                 unsigned int count);

#define FILTER_RANK 6

struct filter_state {
    float state[4 * FILTER_RANK];
};

struct filter_exec {
    float *pInput;
    float *pOutput;
    float *pState;
    float *pCoefficients;
    short count;
};

static void filter_apply_inner(struct filter_exec *pExecState);

// b0, b1, b2, a1, a2
static float filter_coefficients[5 * FILTER_RANK] = {
    // 1
    0.9896377216939719, -1.8465991024871007, 0.9896377216939719,
    1.8523560527906264, -0.9894985766980938,
    // 2
    1, -1.865934434397125, 1, 1.8600773023590333, -0.9897768891698523,
    // // 3
    // 0.2646673220118716, 0.5293346440237432, 0.2646673220118716,
    // 0.11488191101650495, -0.1741032793251565,
    // // 4
    // 1, -2, 1, 1.998958292735337, -0.9989588354706981,
    // 3
    0.11985254882399551, 0.23970509764799103, 0.11985254882399551,
    0.7422952554036788, -0.16765361628847475,
    // 4
    0.125, 0.25, 0.125, 0.979741243381584, -0.544364809428655,
    // 5
    1, -2, 1, 1.9972734230868527, -0.9972756020097775,
    // 6
    1, -2, 1, 1.9988734118111036, -0.9988755828693001};

struct filter_state *filter_state_create(void) {
    struct filter_state *result =
        (struct filter_state *)malloc(sizeof(struct filter_state));
    filter_state_reset(result);
    return result;
}

void filter_state_destroy(struct filter_state *pObject) { free(pObject); }

void filter_state_reset(struct filter_state *pThis) {
    memset(&pThis->state, 0, sizeof(pThis->state));
}

int filter_apply(struct filter_state *pThis, float *pInput, float *pOutput,
                 unsigned int count) {
    struct filter_exec executionState;
    if (!count) return 0;
    executionState.pInput = pInput;
    executionState.pOutput = pOutput;
    executionState.count = count;
    executionState.pState = pThis->state;
    executionState.pCoefficients = filter_coefficients;
    for (int i = 0; i < FILTER_RANK; i++) {
        filter_apply_inner(&executionState);
        executionState.pInput = executionState.pOutput;
    }
    return count;
}

static void filter_apply_inner(struct filter_exec *pExecState) {
    float x0;
    float x1 = pExecState->pState[0];
    float x2 = pExecState->pState[1];
    float y1 = pExecState->pState[2];
    float y2 = pExecState->pState[3];

    float b0 = *(pExecState->pCoefficients++);
    float b1 = *(pExecState->pCoefficients++);
    float b2 = *(pExecState->pCoefficients++);
    float a1 = *(pExecState->pCoefficients++);
    float a2 = *(pExecState->pCoefficients++);

    float *pInput = pExecState->pInput;
    float *pOutput = pExecState->pOutput;
    short count = pExecState->count;
    float accumulator;

    while (count--) {
        x0 = *(pInput++);

        accumulator = x2 * b2;
        accumulator += x1 * b1;
        accumulator += x0 * b0;

        x2 = x1;
        x1 = x0;

        accumulator += y2 * a2;
        accumulator += y1 * a1;

        y2 = y1;
        y1 = accumulator;

        *(pOutput++) = accumulator;
    }

    *(pExecState->pState++) = x1;
    *(pExecState->pState++) = x2;
    *(pExecState->pState++) = y1;
    *(pExecState->pState++) = y2;
}

int main() {
    std::vector<float> input_buffer;
    std::ifstream ifs{"ecg.txt"};
    float val;
    while (ifs >> val) {
        if (val != val) continue;
        input_buffer.push_back(val);
    }
    std::cout << "Total numbers: " << input_buffer.size() << std::endl;
    std::vector<float> output_buffer;
    output_buffer.resize(input_buffer.size());

    struct filter_state *filter = filter_state_create();
    filter_state_reset(filter);
    filter_apply(filter, input_buffer.data(), output_buffer.data(),
                 input_buffer.size());
    filter_state_destroy(filter);
    std::ofstream ofs{"ecg_processed.txt"};
    for (auto &it : output_buffer) {
        ofs << std::fixed << std::setprecision(2) << it << std::endl;
    }
}
