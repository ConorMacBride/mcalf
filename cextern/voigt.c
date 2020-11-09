#include <math.h>

__declspec(dllexport) double func(int n, double args[]) {
    return exp( - pow(args[0], 2.0) / (2.0 * pow(args[2], 2.0))) / (pow(args[3], 2.0) + pow(args[1] - args[0], 2.0));
}
