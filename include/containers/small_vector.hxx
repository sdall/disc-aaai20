#pragma once

#include "SmallVector.h"

namespace sd
{
using small_vector_base = llvm_vecsmall::SmallVectorBase;
template <typename T, size_t N>
using small_vector = llvm_vecsmall::SmallVector<T, N>;
} // namespace sd