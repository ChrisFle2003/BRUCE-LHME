#pragma once

#include <string>

#include "bruce/lhme/formula.hpp"
#include "bruce/lhme/technique.hpp"

namespace bruce::lhme {

// An Engine-owned declaration that one exact Technique request has the same
// result semantics as a registered FormulaSpec.  The Engine compares
// accepted_parameters byte-for-byte; it never parses Technique parameters.
struct ExactFormulaTechniqueCacheBinding {
  TechniqueIdentity technique;
  std::string accepted_parameters;
  FormulaSpec formula;
};

}  // namespace bruce::lhme
