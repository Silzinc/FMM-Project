#include "sample.hpp"
#include "utils.hpp"
#include <optional>

namespace fmm
{
MassSample::MassSample(
  const Vec3 pos,
  const double mass,
  const std::optional<Vec3> prev_pos
)
  : mass(mass)
  , position(pos)
  , prev_pos(prev_pos.value_or(pos))
{}

Vec3
MassSample::speed(const double dt) const
{
  return (position - prev_pos) / dt;
}
}
