#include "PPintrin.h"
// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?

  // All ones
  maskAll = _pp_init_ones();

  // All zeros
  maskIsNegative = _pp_init_ones(0);

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    if(N-i<VECTOR_WIDTH)
      maskAll = _pp_init_ones(N-i);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //

  __pp_vec_float x;
  __pp_vec_int y;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_vec_float one = _pp_vset_float(1.f);
  __pp_vec_float clamp = _pp_vset_float(9.999999f);
  __pp_vec_int zero_int = _pp_vset_int(0);
  __pp_vec_int one_int = _pp_vset_int(1);
  __pp_mask maskAll, maskIsZero, maskIsNotZero, maskClamp;

  // All ones
  maskAll = _pp_init_ones();

  // All zeros
  maskIsZero = _pp_init_ones(0);
  maskClamp = _pp_init_ones(0);

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    if(N-i<VECTOR_WIDTH)
      maskAll = _pp_init_ones(N-i);
    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];
    _pp_vload_int(y, exponents + i, maskAll); // y = exponents[i];

    // Set mask according to predicate
    _pp_veq_int(maskIsZero, y, zero_int, maskAll); // if (y == 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vset_float(result, 1.f, maskIsZero); //   output[i] = 1.f;

    // Inverse maskIsZero to generate "else" mask
    maskIsNotZero = _pp_mask_not(maskIsZero); // } else {
    
    // Execute instruction ("else" clause)
    _pp_vmove_float(result, one, maskIsNotZero); // result = 1.f;

    int count = _pp_cntbits(maskIsNotZero); // number of '1' in maskIsNotZero

    while(count > 0)
    {

      _pp_vmult_float(result, result, x, maskIsNotZero); // result *= x;
      
      _pp_vsub_int(y, y, one_int, maskIsNotZero); // y = y -1;
      
      _pp_vgt_int(maskIsNotZero, y, zero_int, maskIsNotZero); // if(y > 0)
      
      count = _pp_cntbits(maskIsNotZero); // number of '1' in maskIsNotZero
    
    }

    _pp_vgt_float(maskClamp, result, clamp, maskAll); // if(result > 9.999999f)

    _pp_vmove_float(result, clamp, maskClamp); // result = 9.999999f;

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
    
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  __pp_vec_float x;
  __pp_mask maskAll;
  float result[VECTOR_WIDTH];
  float output = 0.f;

  // All ones
  maskAll = _pp_init_ones();

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];
    for(int j=0;j<VECTOR_WIDTH/2;j++)
    {
      _pp_hadd_float(x,x);
      _pp_interleave_float(x,x);

    }

    // Write results back to memory
    _pp_vstore_float(result, x, maskAll);
    output += result[0];
  }

  return output;
}